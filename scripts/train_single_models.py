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


# ==================== PtychoNN 模型 ====================
class PtychoNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.dec1_1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.dec1_2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.dec1_3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.out1 = nn.Conv2d(32, 1, 3, padding='same')
        
        self.dec2_1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.dec2_2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.dec2_3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.out2 = nn.Conv2d(32, 1, 3, padding='same')
    
    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        encoded = self.enc3(x)
        
        x1 = self.dec1_1(encoded)
        x1 = self.dec1_2(x1)
        x1 = self.dec1_3(x1)
        out1 = self.out1(x1)
        
        x2 = self.dec2_1(encoded)
        x2 = self.dec2_2(x2)
        x2 = self.dec2_3(x2)
        out2 = self.out2(x2)
        
        return out1, out2


# ==================== TransUNet 模型 ====================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=1, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

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

class Mlp(nn.Module):
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

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.3, attn_drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TransUNet(nn.Module):
    def __init__(self, img_size=64, in_channels=1, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding='same'),
            nn.ReLU(),
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
            nn.Conv2d(256, 128, 3, padding='same'),
            nn.ReLU(),
        )
        self.dec_amp_fuse1 = nn.Conv2d(128 + 128, 128, 3, padding='same')
        self.dec_amp_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding='same'),
            nn.ReLU(),
        )
        self.dec_amp_fuse2 = nn.Conv2d(64 + 64, 64, 3, padding='same')
        self.dec_amp_up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding='same'),
            nn.ReLU(),
        )
        self.outc_amp = nn.Conv2d(32, 1, 3, padding='same')
        
        self.dec_phase_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding='same'),
            nn.ReLU(),
        )
        self.dec_phase_fuse1 = nn.Conv2d(128 + 128, 128, 3, padding='same')
        self.dec_phase_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding='same'),
            nn.ReLU(),
        )
        self.dec_phase_fuse2 = nn.Conv2d(64 + 64, 64, 3, padding='same')
        self.dec_phase_up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding='same'),
            nn.ReLU(),
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
        amp_out = self.outc_amp(x1)
        
        x2 = self.dec_phase_up1(x_bottleneck)
        x2 = torch.cat([x2, s2], dim=1)
        x2 = self.dec_phase_fuse1(x2)
        x2 = self.dec_phase_up2(x2)
        x2 = torch.cat([x2, s1], dim=1)
        x2 = self.dec_phase_fuse2(x2)
        x2 = self.dec_phase_up3(x2)
        phase_out = self.outc_phase(x2)
        
        return amp_out, phase_out


# ==================== 数据集 ====================
class SingleDiffractionDataset(Dataset):
    def __init__(self, data_root, split='train'):
        self.split = split
        self.input_dir = os.path.join(data_root, split, 'inputs')
        self.label_dir = os.path.join(data_root, split, 'labels')
        self.input_files = sorted(glob.glob(os.path.join(self.input_dir, 'input_*.npy')))

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = self.input_files[idx]
        base_name = os.path.basename(input_path).replace('input_', '')
        input_array = np.load(input_path).astype(np.float32)
        input_tensor = torch.from_numpy(input_array).unsqueeze(0)
        
        label_path = os.path.join(self.label_dir, f'label_{base_name}')
        label_array = np.load(label_path).astype(np.float32)
        label_tensor = torch.from_numpy(label_array)
        
        return input_tensor, label_tensor


# ==================== 训练函数 ====================
def train_model(model, model_name, train_loader, criterion, optimizer, scheduler, epochs=100, device='cuda'):
    os.makedirs(f'{BASE_DIR}/models', exist_ok=True)
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
            torch.save(model.state_dict(), f'{BASE_DIR}/models/best_{model_name.lower()}_single.pth')
        
        print(f"{model_name} - Epoch [{epoch+1:3d}/{epochs}], Loss: {avg_loss:.6f}")
    
    torch.save(model.state_dict(), f'{BASE_DIR}/models/{model_name.lower()}_weights_single.pth')
    print(f"\n{model_name} 训练完成！")
    return train_losses


# ==================== 测试函数 ====================
def test_model(model, test_loader, device='cuda'):
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
    os.makedirs(f'{BASE_DIR}/models', exist_ok=True)
    os.makedirs(f'{BASE_DIR}/train_results', exist_ok=True)
    os.makedirs(f'{BASE_DIR}/test_results', exist_ok=True)
    
    print("加载数据集...")
    train_dataset = SingleDiffractionDataset(f'{BASE_DIR}/data_single', split='train')
    test_dataset = SingleDiffractionDataset(f'{BASE_DIR}/data_single', split='test')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    print(f"训练样本数: {len(train_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")
    
    x, y = train_dataset[0]
    print(f"输入尺寸: {x.shape}, 输出尺寸: {y.shape}")
    
    EPOCHS = 120
    LR = 1e-4
    
    # 训练 PtychoNN
    print("\n创建 PtychoNN 模型...")
    model1 = PtychoNN().to(device)
    criterion = nn.MSELoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=LR)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, 'min', factor=0.5, patience=5, min_lr=1e-6)
    losses1 = train_model(model1, "PtychoNN", train_loader, criterion, optimizer1, scheduler1, EPOCHS)
    
    # 训练 TransUNet
    print("\n创建 TransUNet 模型...")
    model2 = TransUNet(
        img_size=64,
        in_channels=1,
        embed_dim=256,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0
    ).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=LR)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 'min', factor=0.5, patience=5, min_lr=1e-6)
    losses2 = train_model(model2, "TransUNet", train_loader, criterion, optimizer2, scheduler2, EPOCHS)
    
    # 绘制损失曲线对比
    print("\n绘制训练损失对比...")
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, EPOCHS+1), losses1, 'b-', linewidth=2, label='PtychoNN')
    plt.plot(range(1, EPOCHS+1), losses2, 'r-', linewidth=2, label='TransUNet')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training Loss Comparison (Single Diffraction)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{BASE_DIR}/train_results/loss_comparison_single.png', dpi=300, bbox_inches='tight')
    print("损失对比图已保存")
    
    # 在测试集上评估
    print("\n在测试集上评估...")
    pred1_I, pred1_phi, Y1_I, Y1_phi, mse1_I, mse1_phi, mae1_I, mae1_phi = test_model(model1, test_loader)
    pred2_I, pred2_phi, Y2_I, Y2_phi, mse2_I, mse2_phi, mae2_I, mae2_phi = test_model(model2, test_loader)
    
    print("\n" + "=" * 80)
    print("=== 测试集评估结果 ===")
    print("=" * 80)
    print(f"PtychoNN - 振幅 MSE: {mse1_I:.6f}, 相位 MSE: {mse1_phi:.6f}")
    print(f"PtychoNN - 振幅 MAE: {mae1_I:.6f}, 相位 MAE: {mae1_phi:.6f}")
    print(f"TransUNet - 振幅 MSE: {mse2_I:.6f}, 相位 MSE: {mse2_phi:.6f}")
    print(f"TransUNet - 振幅 MAE: {mae2_I:.6f}, 相位 MAE: {mae2_phi:.6f}")
    print("=" * 80)
    
    # 保存测试结果
    np.savez(f'{BASE_DIR}/test_results/single_predictions.npz',
             pred1_I=pred1_I, pred1_phi=pred1_phi,
             pred2_I=pred2_I, pred2_phi=pred2_phi,
             Y_I=Y1_I, Y_phi=Y1_phi)
    
    with open(f'{BASE_DIR}/test_results/single_evaluation.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("单一衍射图案模型测试结果\n")
        f.write("=" * 80 + "\n\n")
        f.write("PtychoNN\n")
        f.write(f"  振幅 MSE: {mse1_I:.6f}\n")
        f.write(f"  相位 MSE: {mse1_phi:.6f}\n")
        f.write(f"  振幅 MAE: {mae1_I:.6f}\n")
        f.write(f"  相位 MAE: {mae1_phi:.6f}\n\n")
        f.write("TransUNet\n")
        f.write(f"  振幅 MSE: {mse2_I:.6f}\n")
        f.write(f"  相位 MSE: {mse2_phi:.6f}\n")
        f.write(f"  振幅 MAE: {mae2_I:.6f}\n")
        f.write(f"  相位 MAE: {mae2_phi:.6f}\n")
    
    # 保存测试集对比图片
    print("\n保存测试集对比图片...")
    num_samples_to_save = 5
    for i in range(num_samples_to_save):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        axes[0, 0].imshow(Y1_I[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title('Ground Truth (Amplitude)', fontsize=10)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(pred1_I[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title('PtychoNN (Amplitude)', fontsize=10)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(pred2_I[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, 2].set_title('TransUNet (Amplitude)', fontsize=10)
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(np.abs(pred1_I[i, 0] - Y1_I[i, 0]), cmap='jet', vmin=0, vmax=0.2)
        axes[0, 3].set_title('PtychoNN Error', fontsize=10)
        axes[0, 3].axis('off')
        
        axes[1, 0].imshow(Y1_phi[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title('Ground Truth (Phase)', fontsize=10)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(pred1_phi[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title('PtychoNN (Phase)', fontsize=10)
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(pred2_phi[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, 2].set_title('TransUNet (Phase)', fontsize=10)
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(np.abs(pred1_phi[i, 0] - Y1_phi[i, 0]), cmap='jet', vmin=0, vmax=0.2)
        axes[1, 3].set_title('PtychoNN Phase Error', fontsize=10)
        axes[1, 3].axis('off')
        
        plt.suptitle(f'Single Diffraction Test Sample {i+1}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{BASE_DIR}/test_results/single_sample_{i+1}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"已保存 {num_samples_to_save} 组测试集对比图片")
    print("\n单一衍射图案模型训练和测试完成！")