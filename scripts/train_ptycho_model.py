import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm

BASE_DIR = 'd:/BaiduSyncdisk/second_term/Uptynet/article_experiment'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ==================== ResidualBlock ====================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ==================== Attention ====================
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.3, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ==================== MLP ====================
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ==================== TransformerBlock ====================
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.3, attn_drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ==================== TransUNet-Ptycho 模型 (基于 train_ptycho_three_models.py) ====================
class TransUNetPtycho(nn.Module):
    def __init__(self, img_size=64, in_channels=100, embed_dim=256, depth=2, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.enc1 = nn.Sequential(
            ResidualBlock(in_channels, 64),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2)
        )
        self.enc3 = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.MaxPool2d(2)
        )
        self.proj_to_embed = nn.Conv2d(256, embed_dim, 1)
        self.proj_from_embed = nn.Conv2d(embed_dim, 256, 1)
        self.transformer_norm = nn.LayerNorm(embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                             drop=0.05, attn_drop=0.025)
            for _ in range(depth)
        ])
        self.dec_amp_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock(256, 256),
        )
        self.dec_amp_fuse1 = ResidualBlock(256 + 128, 128)
        self.dec_amp_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock(128, 128),
        )
        self.dec_amp_fuse2 = ResidualBlock(128 + 64, 64)
        self.dec_amp_up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock(64, 64),
        )
        # 额外上采样到128x128
        self.dec_amp_up4 = nn.Sequential(
            nn.Upsample(size=(218, 218), mode='bilinear', align_corners=True),
            ResidualBlock(64, 32),
        )
        self.outc_amp = nn.Conv2d(32, 1, 3, padding='same')
        
        self.dec_phase_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock(256, 256),
        )
        self.dec_phase_fuse1 = ResidualBlock(256 + 128, 128)
        self.dec_phase_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock(128, 128),
        )
        self.dec_phase_fuse2 = ResidualBlock(128 + 64, 64)
        self.dec_phase_up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock(64, 64),
        )
        # 额外上采样到128x128
        self.dec_phase_up4 = nn.Sequential(
            nn.Upsample(size=(218, 218), mode='bilinear', align_corners=True),
            ResidualBlock(64, 32),
        )
        self.outc_phase = nn.Conv2d(32, 1, 3, padding='same')
    
    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        B, C, H, W = s3.shape
        embed = self.proj_to_embed(s3)
        tokens = embed.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.transformer_norm(tokens)
        x_t = tokens.transpose(1, 2).reshape(B, -1, H, W)
        x_t = self.proj_from_embed(x_t)
        x_bottleneck = x_t + s3
        
        x1 = self.dec_amp_up1(x_bottleneck)
        x1 = torch.cat([x1, s2], dim=1)
        x1 = self.dec_amp_fuse1(x1)
        x1 = self.dec_amp_up2(x1)
        x1 = torch.cat([x1, s1], dim=1)
        x1 = self.dec_amp_fuse2(x1)
        x1 = self.dec_amp_up3(x1)
        x1 = self.dec_amp_up4(x1)
        amp_out = self.outc_amp(x1)
        
        x2 = self.dec_phase_up1(x_bottleneck)
        x2 = torch.cat([x2, s2], dim=1)
        x2 = self.dec_phase_fuse1(x2)
        x2 = self.dec_phase_up2(x2)
        x2 = torch.cat([x2, s1], dim=1)
        x2 = self.dec_phase_fuse2(x2)
        x2 = self.dec_phase_up3(x2)
        x2 = self.dec_phase_up4(x2)
        phase_out = self.outc_phase(x2)
        
        return amp_out, phase_out


# ==================== 数据集 ====================
class PtychoDataset(Dataset):
    def __init__(self, data_root, split='train', output_size=218):
        self.split = split
        self.input_dir = os.path.join(data_root, split, 'inputs')
        self.label_dir = os.path.join(data_root, split, 'labels')
        self.input_files = sorted(glob.glob(os.path.join(self.input_dir, 'input_*.npy')))
        self.output_size = output_size

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = self.input_files[idx]
        base_name = os.path.basename(input_path).replace('input_', '')
        input_array = np.load(input_path).astype(np.float32)
        input_tensor = torch.from_numpy(input_array)
        
        label_path = os.path.join(self.label_dir, f'label_{base_name}')
        label_array = np.load(label_path).astype(np.float32)
        label_tensor = torch.from_numpy(label_array)
        
        if label_tensor.shape[1] != self.output_size or label_tensor.shape[2] != self.output_size:
            label_tensor = torch.nn.functional.interpolate(
                label_tensor.unsqueeze(0), 
                size=(self.output_size, self.output_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        return input_tensor, label_tensor


# ==================== 训练函数 ====================
def train_ptycho_model(model, model_name, train_loader, criterion, optimizer, scheduler, epochs=80, device='cuda'):
    os.makedirs(f'{BASE_DIR}/models_overlap025', exist_ok=True)
    print(f"\n训练 {model_name}...")
    print("=" * 80)
    train_losses = []
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')):
            x = x.to(device)
            y_i = y[:, 0:1, :, :].to(device)
            y_phi = y[:, 1:2, :, :].to(device)
            optimizer.zero_grad()
            pred_i, pred_phi = model(x)
            loss_i = criterion(pred_i, y_i)
            loss_phi = criterion(pred_phi, y_phi)
            loss = loss_i + loss_phi
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'{BASE_DIR}/models_overlap025/best_{model_name.lower()}_ptycho.pth')
        
        print(f"{model_name} - Epoch [{epoch+1:3d}/{epochs}], Loss: {avg_loss:.6f}")
    
    torch.save(model.state_dict(), f'{BASE_DIR}/models_overlap025/{model_name.lower()}_ptycho_weights.pth')
    print(f"\n{model_name} 训练完成！")
    return train_losses

# ==================== 测试函数 ====================
def test_ptycho_model(model, test_loader, device='cuda'):
    model.eval()
    all_pred_I = []
    all_pred_phi = []
    all_Y_I = []
    all_Y_phi = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="测试中"):
            x = x.to(device)
            y_i = y[:, 0:1, :, :]
            y_phi = y[:, 1:2, :, :]
            pred_i, pred_phi = model(x)
            all_pred_I.append(pred_i.cpu().numpy())
            all_pred_phi.append(pred_phi.cpu().numpy())
            all_Y_I.append(y_i.numpy())
            all_Y_phi.append(y_phi.numpy())
    
    pred_I = np.concatenate(all_pred_I, axis=0)
    pred_phi = np.concatenate(all_pred_phi, axis=0)
    Y_I = np.concatenate(all_Y_I, axis=0)
    Y_phi = np.concatenate(all_Y_phi, axis=0)
    
    mse_I = np.mean((pred_I - Y_I)**2)
    mse_phi = np.mean((pred_phi - Y_phi)**2)
    mae_I = np.mean(np.abs(pred_I - Y_I))
    mae_phi = np.mean(np.abs(pred_phi - Y_phi))
    
    return pred_I, pred_phi, Y_I, Y_phi, mse_I, mse_phi, mae_I, mae_phi


# ==================== 主函数 ====================
if __name__ == '__main__':
    os.makedirs(f'{BASE_DIR}/models_overlap025', exist_ok=True)
    os.makedirs(f'{BASE_DIR}/train_results_overlap025', exist_ok=True)
    os.makedirs(f'{BASE_DIR}/test_results_overlap025', exist_ok=True)
    
    print("加载叠层成像数据集...")
    train_dataset = PtychoDataset(f'{BASE_DIR}/data_ptycho_overlap025', split='train')
    test_dataset = PtychoDataset(f'{BASE_DIR}/data_ptycho_overlap025', split='test')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    print(f"训练样本数: {len(train_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")
    
    x, y = train_dataset[0]
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {y.shape}")
    
    num_channels = x.shape[0]
    output_size = y.shape[2]
    print(f"检测到输入通道数: {num_channels}")
    print(f"检测到输出尺寸: {output_size}×{output_size}")
    
    EPOCHS = 120
    LR = 1e-4
    
    model = TransUNetPtycho(
        img_size=64,
        in_channels=num_channels,
        embed_dim=256,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-6)
    
    losses = train_ptycho_model(model, "TransUNet-Ptycho", train_loader, criterion, optimizer, scheduler, EPOCHS)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, EPOCHS+1), losses, 'g-', linewidth=2, label='TransUNet-Ptycho')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training Loss (Ptycho)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{BASE_DIR}/train_results_overlap025/loss_ptycho.png', dpi=300, bbox_inches='tight')
    
    pred_I, pred_phi, Y_I, Y_phi, mse_I, mse_phi, mae_I, mae_phi = test_ptycho_model(model, test_loader)
    
    print("\n" + "=" * 80)
    print("=== TransUNet-Ptycho 测试集评估结果 ===")
    print("=" * 80)
    print(f"振幅 MSE: {mse_I:.6f}, 相位 MSE: {mse_phi:.6f}")
    print(f"振幅 MAE: {mae_I:.6f}, 相位 MAE: {mae_phi:.6f}")
    print("=" * 80)
    
    np.savez(f'{BASE_DIR}/test_results_overlap025/ptycho_predictions.npz',
             pred_I=pred_I, pred_phi=pred_phi, Y_I=Y_I, Y_phi=Y_phi)
    
    with open(f'{BASE_DIR}/test_results_overlap025/ptycho_evaluation.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TransUNet-Ptycho 测试结果\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"输入通道数: {num_channels}\n")
        f.write(f"输出尺寸: {output_size}×{output_size}\n\n")
        f.write(f"振幅 MSE: {mse_I:.6f}\n")
        f.write(f"相位 MSE: {mse_phi:.6f}\n")
        f.write(f"振幅 MAE: {mae_I:.6f}\n")
        f.write(f"相位 MAE: {mae_phi:.6f}\n")
    
    # 保存测试集对比图片
    print("\n保存测试集对比图片...")
    num_samples_to_save = 5
    for i in range(num_samples_to_save):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(Y_I[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title('Ground Truth (Amplitude)', fontsize=10)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(pred_I[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title('TransUNet-Ptycho (Amplitude)', fontsize=10)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(np.abs(pred_I[i, 0] - Y_I[i, 0]), cmap='jet', vmin=0, vmax=0.2)
        axes[0, 2].set_title('Amplitude Error', fontsize=10)
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(Y_phi[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title('Ground Truth (Phase)', fontsize=10)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(pred_phi[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title('TransUNet-Ptycho (Phase)', fontsize=10)
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(np.abs(pred_phi[i, 0] - Y_phi[i, 0]), cmap='jet', vmin=0, vmax=0.2)
        axes[1, 2].set_title('Phase Error', fontsize=10)
        axes[1, 2].axis('off')
        
        plt.suptitle(f'TransUNet-Ptycho Test Sample {i+1}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{BASE_DIR}/test_results_overlap025/ptycho_sample_{i+1}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"已保存 {num_samples_to_save} 组测试集对比图片")
    print("\nTransUNet-Ptycho 训练和测试完成！")