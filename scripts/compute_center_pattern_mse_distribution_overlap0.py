import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

BASE_DIR = 'd:/BaiduSyncdisk/second_term/Uptynet/article_experiment'
OUTPUT_DIR = 'd:/BaiduSyncdisk/second_term/Uptynet/article_experiment/overlap_comparison/statistics'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class PtychoDataset(Dataset):
    def __init__(self, data_root, split='train', output_size=272):
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
        
        return input_tensor, label_tensor

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

def create_ptycho_model(output_size):
    class TransUNetPtycho(nn.Module):
        def __init__(self, img_size=64, in_channels=100, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4.0):
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
            self.dec_amp_up4 = nn.Sequential(
                nn.Upsample(size=(output_size, output_size), mode='bilinear', align_corners=True),
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
            self.dec_phase_up4 = nn.Sequential(
                nn.Upsample(size=(output_size, output_size), mode='bilinear', align_corners=True),
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
    
    return TransUNetPtycho()

def compute_mse(pred, gt):
    return np.mean((pred - gt) ** 2)

def generate_mse_distribution_plot(mse_data, save_dir):
    colors = ['#3498db', '#e74c3c']
    labels = ['PtychoNN', 'Ptycho-TransUNet']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    axes[0].hist(mse_data['ptychonn_amp'], bins=50, alpha=0.5, color=colors[0], label=f'{labels[0]} Amp', edgecolor='black')
    axes[0].hist(mse_data['ptycho_amp'], bins=50, alpha=0.5, color=colors[1], label=f'{labels[1]} Amp', edgecolor='black')
    
    axes[0].axvline(np.mean(mse_data['ptychonn_amp']), color=colors[0], linestyle='--', linewidth=2, 
                    label=f'{labels[0]} Mean: {np.mean(mse_data["ptychonn_amp"]):.4f}')
    axes[0].axvline(np.mean(mse_data['ptycho_amp']), color=colors[1], linestyle='--', linewidth=2,
                    label=f'{labels[1]} Mean: {np.mean(mse_data["ptycho_amp"]):.4f}')
    
    axes[0].set_xlabel('MSE')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Amplitude MSE Distribution (64×64 Region)')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(mse_data['ptychonn_phase'], bins=50, alpha=0.5, color=colors[0], label=f'{labels[0]} Phase', edgecolor='black')
    axes[1].hist(mse_data['ptycho_phase'], bins=50, alpha=0.5, color=colors[1], label=f'{labels[1]} Phase', edgecolor='black')
    
    axes[1].axvline(np.mean(mse_data['ptychonn_phase']), color=colors[0], linestyle='--', linewidth=2,
                    label=f'{labels[0]} Mean: {np.mean(mse_data["ptychonn_phase"]):.4f}')
    axes[1].axvline(np.mean(mse_data['ptycho_phase']), color=colors[1], linestyle='--', linewidth=2,
                    label=f'{labels[1]} Mean: {np.mean(mse_data["ptycho_phase"]):.4f}')
    
    axes[1].set_xlabel('MSE')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Phase MSE Distribution (64×64 Region)')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('MSE Distribution Comparison: PtychoNN vs Ptycho-TransUNet\n(Overlap=0, 1250 Samples, Center Pattern #49)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, 'mse_distribution_overlap0_ptychonn_vs_ptycho.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存 MSE 分布图: mse_distribution_overlap0_ptychonn_vs_ptycho.png")

def main():
    output_size = 272
    step_size = 23
    patch_size = 64
    grid_size_x = 10
    grid_size_y = 10
    center_pattern_idx = 49
    
    ptycho_model = create_ptycho_model(output_size).to(device)
    ptycho_path = f"{BASE_DIR}/models_overlap0/best_transunet-ptycho_ptycho.pth"
    if os.path.exists(ptycho_path):
        ptycho_model.load_state_dict(torch.load(ptycho_path, map_location=device, weights_only=True))
    ptycho_model.eval()
    print(f"加载 Ptycho-TransUNet 模型: {ptycho_path}")
    
    ptycho_nn = PtychoNN().to(device)
    ptycho_nn_path = f"{BASE_DIR}/models/best_ptychonn_single.pth"
    if os.path.exists(ptycho_nn_path):
        ptycho_nn.load_state_dict(torch.load(ptycho_nn_path, map_location=device, weights_only=True))
    ptycho_nn.eval()
    print(f"加载 PtychoNN 模型: {ptycho_nn_path}")
    
    test_dataset = PtychoDataset(f'{BASE_DIR}/data_ptycho_overlap0', split='test', output_size=output_size)
    num_samples = len(test_dataset)
    print(f"\n测试集样本数: {num_samples}")
    print(f"每个样本衍射图案数: 100")
    print(f"选取中间衍射图案索引: {center_pattern_idx}")
    
    ptychonn_amp_mse = []
    ptychonn_phase_mse = []
    ptycho_amp_mse = []
    ptycho_phase_mse = []
    
    print("\n" + "="*60)
    print("开始处理 1250 个样本 (Overlap=0)...")
    print("="*60)
    
    for sample_idx in tqdm(range(num_samples), desc='Processing samples'):
        x, y = test_dataset[sample_idx]
        
        row = center_pattern_idx // grid_size_x
        col = center_pattern_idx % grid_size_x
        
        y_start = (grid_size_y - row - 1) * step_size
        x_start = col * step_size
        y_end = y_start + patch_size
        x_end = x_start + patch_size
        
        gt_amp = y[0, y_start:y_end, x_start:x_end].numpy()
        gt_phase = y[1, y_start:y_end, x_start:x_end].numpy()
        
        single_input = x[center_pattern_idx:center_pattern_idx+1].unsqueeze(0).to(device)
        
        with torch.no_grad():
            ptycho_nn_amp, ptycho_nn_phase = ptycho_nn(single_input)
        
        ptycho_nn_amp_np = ptycho_nn_amp.squeeze().cpu().numpy()
        ptycho_nn_phase_np = ptycho_nn_phase.squeeze().cpu().numpy()
        
        x_batch = x.unsqueeze(0).to(device)
        with torch.no_grad():
            ptycho_full_amp, ptycho_full_phase = ptycho_model(x_batch)
        
        ptycho_full_amp_np = ptycho_full_amp.squeeze().cpu().numpy()
        ptycho_full_phase_np = ptycho_full_phase.squeeze().cpu().numpy()
        
        ptycho_crop_amp = ptycho_full_amp_np[y_start:y_end, x_start:x_end]
        ptycho_crop_phase = ptycho_full_phase_np[y_start:y_end, x_start:x_end]
        
        mse_ptychonn_amp = compute_mse(ptycho_nn_amp_np, gt_amp)
        mse_ptychonn_phase = compute_mse(ptycho_nn_phase_np, gt_phase)
        mse_ptycho_amp = compute_mse(ptycho_crop_amp, gt_amp)
        mse_ptycho_phase = compute_mse(ptycho_crop_phase, gt_phase)
        
        ptychonn_amp_mse.append(mse_ptychonn_amp)
        ptychonn_phase_mse.append(mse_ptychonn_phase)
        ptycho_amp_mse.append(mse_ptycho_amp)
        ptycho_phase_mse.append(mse_ptycho_phase)
    
    mse_data = {
        'ptychonn_amp': np.array(ptychonn_amp_mse),
        'ptychonn_phase': np.array(ptychonn_phase_mse),
        'ptycho_amp': np.array(ptycho_amp_mse),
        'ptycho_phase': np.array(ptycho_phase_mse)
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    np.savez(os.path.join(OUTPUT_DIR, 'mse_distribution_overlap0_ptychonn_vs_ptycho.npz'), **mse_data)
    
    with open(os.path.join(OUTPUT_DIR, 'mse_distribution_overlap0_ptychonn_vs_ptycho.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("MSE Statistics: PtychoNN vs Ptycho-TransUNet\n")
        f.write("="*60 + "\n")
        f.write(f"Dataset: Overlap=0\n")
        f.write(f"Samples: {num_samples}\n")
        f.write(f"Center Pattern Index: {center_pattern_idx}\n")
        f.write(f"Patch Size: {patch_size}×{patch_size}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("PtychoNN:\n")
        f.write(f"  Amplitude MSE: mean={np.mean(mse_data['ptychonn_amp']):.6f}, std={np.std(mse_data['ptychonn_amp']):.6f}\n")
        f.write(f"  Phase MSE:     mean={np.mean(mse_data['ptychonn_phase']):.6f}, std={np.std(mse_data['ptychonn_phase']):.6f}\n")
        f.write("\nPtycho-TransUNet:\n")
        f.write(f"  Amplitude MSE: mean={np.mean(mse_data['ptycho_amp']):.6f}, std={np.std(mse_data['ptycho_amp']):.6f}\n")
        f.write(f"  Phase MSE:     mean={np.mean(mse_data['ptycho_phase']):.6f}, std={np.std(mse_data['ptycho_phase']):.6f}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("Comparison:\n")
        f.write(f"  Amplitude MSE Improvement: {(np.mean(mse_data['ptychonn_amp']) - np.mean(mse_data['ptycho_amp'])) / np.mean(mse_data['ptychonn_amp']) * 100:.2f}%\n")
        f.write(f"  Phase MSE Improvement:     {(np.mean(mse_data['ptychonn_phase']) - np.mean(mse_data['ptycho_phase'])) / np.mean(mse_data['ptychonn_phase']) * 100:.2f}%\n")
    
    generate_mse_distribution_plot(mse_data, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("统计结果: PtychoNN vs Ptycho-TransUNet (Overlap=0)")
    print("="*60)
    print("PtychoNN:")
    print(f"  Amplitude MSE: mean={np.mean(mse_data['ptychonn_amp']):.6f}, std={np.std(mse_data['ptychonn_amp']):.6f}")
    print(f"  Phase MSE:     mean={np.mean(mse_data['ptychonn_phase']):.6f}, std={np.std(mse_data['ptychonn_phase']):.6f}")
    print("\nPtycho-TransUNet:")
    print(f"  Amplitude MSE: mean={np.mean(mse_data['ptycho_amp']):.6f}, std={np.std(mse_data['ptycho_amp']):.6f}")
    print(f"  Phase MSE:     mean={np.mean(mse_data['ptycho_phase']):.6f}, std={np.std(mse_data['ptycho_phase']):.6f}")
    print("\n" + "="*60)
    print("改进百分比:")
    amp_improvement = (np.mean(mse_data['ptychonn_amp']) - np.mean(mse_data['ptycho_amp'])) / np.mean(mse_data['ptychonn_amp']) * 100
    phase_improvement = (np.mean(mse_data['ptychonn_phase']) - np.mean(mse_data['ptycho_phase'])) / np.mean(mse_data['ptychonn_phase']) * 100
    print(f"  振幅 MSE 改进: {amp_improvement:.2f}%")
    print(f"  相位 MSE 改进: {phase_improvement:.2f}%")
    
    print(f"\n所有结果保存在: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()