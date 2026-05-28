import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset

BASE_DIR = 'd:/BaiduSyncdisk/second_term/Uptynet/article_experiment'
OUTPUT_DIR = 'd:/BaiduSyncdisk/second_term/Uptynet/article_experiment/overlap_comparison/visualizations'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class PtychoDataset(Dataset):
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

def find_sample_with_decreasing_mse():
    mse_files = [
        ('mse_distribution_overlap0_ptychonn_vs_ptycho.npz', 'Overlap=0'),
        ('mse_distribution_overlap025_ptychonn_vs_ptycho.npz', 'Overlap=0.25'),
        ('mse_distribution_overlap05_ptychonn_vs_ptycho.npz', 'Overlap=0.5'),
        ('mse_distribution_overlap07_ptychonn_vs_ptycho.npz', 'Overlap=0.7')
    ]
    
    mse_data = []
    for filename, label in mse_files:
        filepath = os.path.join(BASE_DIR, 'overlap_comparison/statistics', filename)
        if os.path.exists(filepath):
            data = np.load(filepath)
            mse_data.append((label, data['ptycho_amp']))
            print(f"加载 {filename}")
        else:
            print(f"警告: {filename} 不存在")
            return None
    
    for i0 in range(len(mse_data[0][1])):
        mse0 = mse_data[0][1][i0]
        for i25 in range(len(mse_data[1][1])):
            mse25 = mse_data[1][1][i25]
            if mse25 >= mse0:
                continue
            for i5 in range(len(mse_data[2][1])):
                mse5 = mse_data[2][1][i5]
                if mse5 >= mse25:
                    continue
                for i7 in range(len(mse_data[3][1])):
                    mse7 = mse_data[3][1][i7]
                    if mse7 >= mse5:
                        continue
                    print(f"\n找到满足条件的样本:")
                    print(f"  Overlap=0: 样本{i0}, MSE={mse0:.4f}")
                    print(f"  Overlap=0.25: 样本{i25}, MSE={mse25:.4f}")
                    print(f"  Overlap=0.5: 样本{i5}, MSE={mse5:.4f}")
                    print(f"  Overlap=0.7: 样本{i7}, MSE={mse7:.4f}")
                    return [i0, i25, i5, i7]
    
    print("未找到满足条件的样本组合")
    return None

def main():
    sample_indices = find_sample_with_decreasing_mse()
    if sample_indices is None:
        print("无法找到满足条件的样本，使用默认样本")
        sample_indices = [0, 0, 0, 0]
    
    overlap_configs = [
        {'name': 'Overlap=0', 'data_dir': 'data_ptycho_overlap0', 'model_dir': 'models_overlap0', 'output_size': 272, 'step_size': 23, 'sample_idx': sample_indices[0]},
        {'name': 'Overlap=0.25', 'data_dir': 'data_ptycho_overlap025', 'model_dir': 'models_overlap025', 'output_size': 218, 'step_size': 17, 'sample_idx': sample_indices[1]},
        {'name': 'Overlap=0.5', 'data_dir': 'data_ptycho_overlap05', 'model_dir': 'models_overlap05', 'output_size': 164, 'step_size': 11, 'sample_idx': sample_indices[2]},
        {'name': 'Overlap=0.7', 'data_dir': 'data_ptycho', 'model_dir': 'models', 'output_size': 128, 'step_size': 7, 'sample_idx': sample_indices[3]}
    ]
    
    ptycho_nn = PtychoNN().to(device)
    ptycho_nn_path = f"{BASE_DIR}/models/best_ptychonn_single.pth"
    if os.path.exists(ptycho_nn_path):
        ptycho_nn.load_state_dict(torch.load(ptycho_nn_path, map_location=device, weights_only=True))
    ptycho_nn.eval()
    print(f"\n加载 PtychoNN 模型: {ptycho_nn_path}")
    
    center_pattern_idx = 49
    grid_size_x = 10
    grid_size_y = 10
    patch_size = 64
    
    individual_dir = os.path.join(OUTPUT_DIR, 'reconstruction_individual_plots')
    os.makedirs(individual_dir, exist_ok=True)
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    
    mse_values_ptycho = []
    mse_values_ptychonn = []
    
    for i, config in enumerate(overlap_configs):
        print(f"\n处理 {config['name']} (样本{config['sample_idx']})...")
        
        test_dataset = PtychoDataset(f'{BASE_DIR}/{config["data_dir"]}', split='test')
        x, y = test_dataset[config['sample_idx']]
        
        row = center_pattern_idx // grid_size_x
        col = center_pattern_idx % grid_size_x
        
        y_start = (grid_size_y - row - 1) * config['step_size']
        x_start = col * config['step_size']
        y_end = y_start + patch_size
        x_end = x_start + patch_size
        
        if y_end > config['output_size'] or x_end > config['output_size']:
            y_start = config['output_size'] - patch_size
            x_start = config['output_size'] - patch_size
            y_end = config['output_size']
            x_end = config['output_size']
        
        gt_amp = y[0, y_start:y_end, x_start:x_end].numpy()
        gt_phase = y[1, y_start:y_end, x_start:x_end].numpy()
        
        single_input = x[center_pattern_idx:center_pattern_idx+1].unsqueeze(0).to(device)
        with torch.no_grad():
            ptycho_nn_amp, ptycho_nn_phase = ptycho_nn(single_input)
        ptycho_nn_amp_np = ptycho_nn_amp.squeeze().cpu().numpy()
        ptycho_nn_phase_np = ptycho_nn_phase.squeeze().cpu().numpy()
        
        mse_ptychonn = np.mean((ptycho_nn_amp_np - gt_amp)**2)
        mse_values_ptychonn.append(mse_ptychonn)
        
        ptycho_model = create_ptycho_model(config['output_size']).to(device)
        ptycho_path = f"{BASE_DIR}/{config['model_dir']}/best_transunet-ptycho_ptycho.pth"
        if os.path.exists(ptycho_path):
            ptycho_model.load_state_dict(torch.load(ptycho_path, map_location=device, weights_only=True))
        ptycho_model.eval()
        
        x_batch = x.unsqueeze(0).to(device)
        with torch.no_grad():
            ptycho_full_amp, ptycho_full_phase = ptycho_model(x_batch)
        
        ptycho_full_amp_np = ptycho_full_amp.squeeze().cpu().numpy()
        ptycho_full_phase_np = ptycho_full_phase.squeeze().cpu().numpy()
        
        ptycho_crop_amp = ptycho_full_amp_np[y_start:y_end, x_start:x_end]
        ptycho_crop_phase = ptycho_full_phase_np[y_start:y_end, x_start:x_end]
        
        mse_ptycho = np.mean((ptycho_crop_amp - gt_amp)**2)
        mse_values_ptycho.append(mse_ptycho)
        
        fig_gt, ax_gt = plt.subplots(figsize=(5, 5))
        ax_gt.imshow(gt_amp, cmap='gray')
        ax_gt.set_title(f'{config["name"]}\nGT Amp')
        ax_gt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f'{config["name"].replace("=", "_")}_gt_amp.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        fig_ptychonn, ax_ptychonn = plt.subplots(figsize=(5, 5))
        ax_ptychonn.imshow(ptycho_nn_amp_np, cmap='gray')
        ax_ptychonn.set_title(f'{config["name"]}\nPtychoNN Amp\nMSE={mse_ptychonn:.4f}')
        ax_ptychonn.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f'{config["name"].replace("=", "_")}_ptychonn_amp.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        fig_ptycho, ax_ptycho = plt.subplots(figsize=(5, 5))
        ax_ptycho.imshow(ptycho_crop_amp, cmap='gray')
        ax_ptycho.set_title(f'{config["name"]}\nPtycho-TransUNet Amp\nMSE={mse_ptycho:.4f}')
        ax_ptycho.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f'{config["name"].replace("=", "_")}_ptycho_amp.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        axes[i, 0].imshow(gt_amp, cmap='gray')
        axes[i, 0].set_title(f'{config["name"]}\nGT Amp')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(ptycho_nn_amp_np, cmap='gray')
        axes[i, 1].set_title(f'PtychoNN Amp\nMSE={mse_ptychonn:.4f}')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(ptycho_crop_amp, cmap='gray')
        axes[i, 2].set_title(f'Ptycho-TransUNet Amp\nMSE={mse_ptycho:.4f}')
        axes[i, 2].axis('off')
    
    print(f"\n已保存12张单独的图片到: {individual_dir}")
    
    plt.suptitle(f'Reconstruction Comparison: PtychoNN vs Ptycho-TransUNet\n(Center Pattern #49, Amplitude)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'reconstruction_comparison_decreasing_mse.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n重构对比图已保存: {output_path}")
    print(f"PtychoNN MSE: {mse_values_ptychonn[0]:.4f} → {mse_values_ptychonn[1]:.4f} → {mse_values_ptychonn[2]:.4f} → {mse_values_ptychonn[3]:.4f}")
    print(f"Ptycho-TransUNet MSE: {mse_values_ptycho[0]:.4f} → {mse_values_ptycho[1]:.4f} → {mse_values_ptycho[2]:.4f} → {mse_values_ptycho[3]:.4f}")

if __name__ == '__main__':
    main()