import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 8
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd

BASE_DIR = 'd:/BaiduSyncdisk/second_term/Uptynet/article_experiment'
OUTPUT_DIR = 'd:/BaiduSyncdisk/second_term/Uptynet/article_experiment/overlap_comparison'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

OVERLAP_CONFIGS = {
    '0.7': {
        'data_dir': f'{BASE_DIR}/data_ptycho',
        'model_dir': f'{BASE_DIR}/models',
        'overlap': 0.7,
        'output_size': 128,
        'step': 7,
        'color': '#e74c3c'
    },
    '0.5': {
        'data_dir': f'{BASE_DIR}/data_ptycho_overlap05',
        'model_dir': f'{BASE_DIR}/models_overlap05',
        'overlap': 0.5,
        'output_size': 164,
        'step': 11,
        'color': '#3498db'
    },
    '0.25': {
        'data_dir': f'{BASE_DIR}/data_ptycho_overlap025',
        'model_dir': f'{BASE_DIR}/models_overlap025',
        'overlap': 0.25,
        'output_size': 218,
        'step': 17,
        'color': '#2ecc71'
    },
    '0': {
        'data_dir': f'{BASE_DIR}/data_ptycho_overlap0',
        'model_dir': f'{BASE_DIR}/models_overlap0',
        'overlap': 0,
        'output_size': 272,
        'step': 23,
        'color': '#9b59b6'
    }
}

class PtychoDataset(Dataset):
    def __init__(self, data_root, split='train', output_size=128):
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

class TransUNetSingle(nn.Module):
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

def load_models_for_visualization():
    config = OVERLAP_CONFIGS['0.7']
    
    ptycho_model = create_ptycho_model(config['output_size']).to(device)
    ptycho_path = f"{config['model_dir']}/best_transunet-ptycho_ptycho.pth"
    if os.path.exists(ptycho_path):
        ptycho_model.load_state_dict(torch.load(ptycho_path, map_location=device, weights_only=True))
    ptycho_model.eval()
    
    ptycho_nn = PtychoNN().to(device)
    ptycho_nn_path = f"{BASE_DIR}/models/best_ptychonn_single.pth"
    if os.path.exists(ptycho_nn_path):
        ptycho_nn.load_state_dict(torch.load(ptycho_nn_path, map_location=device, weights_only=True))
    ptycho_nn.eval()
    
    transunet_single = TransUNetSingle(
        img_size=64,
        in_channels=1,
        embed_dim=256,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0
    ).to(device)
    transunet_single_path = f"{BASE_DIR}/models/best_transunet_single.pth"
    if os.path.exists(transunet_single_path):
        transunet_single.load_state_dict(torch.load(transunet_single_path, map_location=device, weights_only=True))
    transunet_single.eval()
    
    return ptycho_model, ptycho_nn, transunet_single

def load_ptycho_model(overlap_key):
    config = OVERLAP_CONFIGS[overlap_key]
    ptycho_model = create_ptycho_model(config['output_size']).to(device)
    ptycho_path = f"{config['model_dir']}/best_transunet-ptycho_ptycho.pth"
    if os.path.exists(ptycho_path):
        ptycho_model.load_state_dict(torch.load(ptycho_path, map_location=device, weights_only=True))
    ptycho_model.eval()
    return ptycho_model

def calculate_illuminated_region_size(step):
    illuminated_size = step * 9
    return illuminated_size

def generate_comparison_visualization(overlap_key, test_dataset, models, save_dir):
    config = OVERLAP_CONFIGS[overlap_key]
    output_size = config['output_size']
    ptycho_model, ptycho_nn, transunet_single = models
    
    indices = [0, 1, 2, 3]
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    row_labels = [
        'Ground Truth',
        'PtychoNN Prediction',
        'TransUNet (Single) Prediction',
        'TransUNet-Ptycho Prediction'
    ]
    
    center = output_size // 2
    half = 32
    
    for col, idx in enumerate(indices):
        x, y = test_dataset[idx]
        x_batch = x.unsqueeze(0).to(device)
        
        with torch.no_grad():
            ptycho_amp, ptycho_phase = ptycho_model(x_batch)
            
            center_idx = 49
            single_input = x[center_idx:center_idx+1].unsqueeze(0).to(device)
            transunet_amp, transunet_phase = transunet_single(single_input)
            
            ptycho_nn_amp, ptycho_nn_phase = ptycho_nn(single_input)
        
        y_amp = y[0].numpy()
        y_phase = y[1].numpy()
        
        y_amp_center = y_amp[center-half:center+half, center-half:center+half]
        y_phase_center = y_phase[center-half:center+half, center-half:center+half]
        
        ptycho_amp_np = ptycho_amp.squeeze().cpu().numpy()[center-half:center+half, center-half:center+half]
        ptycho_phase_np = ptycho_phase.squeeze().cpu().numpy()[center-half:center+half, center-half:center+half]
        
        vmin_amp = min(y_amp_center.min(), ptycho_amp_np.min())
        vmax_amp = max(y_amp_center.max(), ptycho_amp_np.max())
        vmin_phase = min(y_phase_center.min(), ptycho_phase_np.min())
        vmax_phase = max(y_phase_center.max(), ptycho_phase_np.max())
        
        axes[0, col].imshow(y_amp_center, cmap='gray', vmin=vmin_amp, vmax=vmax_amp)
        axes[0, col].set_title(f'Sample {idx+1}')
        axes[0, col].axis('off')
        
        axes[1, col].imshow(ptycho_nn_amp.squeeze().cpu().numpy(), cmap='gray')
        axes[1, col].axis('off')
        
        axes[2, col].imshow(transunet_amp.squeeze().cpu().numpy(), cmap='gray')
        axes[2, col].axis('off')
        
        axes[3, col].imshow(ptycho_amp_np, cmap='gray', vmin=vmin_amp, vmax=vmax_amp)
        axes[3, col].axis('off')
    
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, fontsize=12, rotation=0, ha='right', va='center')
    
    plt.suptitle(f'Model Comparison (Overlap = {overlap_key}, 64×64 Center Region)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.08, 0, 1, 0.96])
    
    save_path = os.path.join(save_dir, f'comparison_overlap_{overlap_key}_idx49.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存对比图: {save_path}")

def generate_illuminated_region_comparison(test_datasets, models_dict, save_dir):
    overlap_keys = ['0', '0.25', '0.5', '0.7']
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    row_labels = ['GT Amp', 'Pred Amp', 'GT Phase', 'Pred Phase']
    
    for col, overlap_key in enumerate(overlap_keys):
        config = OVERLAP_CONFIGS[overlap_key]
        output_size = config['output_size']
        step = config['step']
        illuminated_size = calculate_illuminated_region_size(step)
        
        test_dataset = test_datasets[overlap_key]
        model = models_dict[overlap_key]
        
        idx = 0
        x, y = test_dataset[idx]
        x_batch = x.unsqueeze(0).to(device)
        
        with torch.no_grad():
            ptycho_amp, ptycho_phase = model(x_batch)
        
        y_amp = y[0].numpy()
        y_phase = y[1].numpy()
        pred_amp = ptycho_amp.squeeze().cpu().numpy()
        pred_phase = ptycho_phase.squeeze().cpu().numpy()
        
        center = output_size // 2
        half_illum = illuminated_size // 2
        
        center_start = center - half_illum
        center_end = center + half_illum
        
        vmin_amp = min(y_amp.min(), pred_amp.min())
        vmax_amp = max(y_amp.max(), pred_amp.max())
        vmin_phase = min(y_phase.min(), pred_phase.min())
        vmax_phase = max(y_phase.max(), pred_phase.max())
        
        axes[0, col].imshow(y_amp, cmap='gray', vmin=vmin_amp, vmax=vmax_amp)
        rect_gt_amp = plt.Rectangle((center_start, center_start), 
                                     illuminated_size, illuminated_size,
                                     fill=False, edgecolor='red', linewidth=2)
        axes[0, col].add_patch(rect_gt_amp)
        axes[0, col].set_title(f'Overlap={overlap_key}\n({output_size}×{output_size})')
        axes[0, col].axis('off')
        
        axes[1, col].imshow(pred_amp, cmap='gray', vmin=vmin_amp, vmax=vmax_amp)
        rect_pred_amp = plt.Rectangle((center_start, center_start), 
                                       illuminated_size, illuminated_size,
                                       fill=False, edgecolor='red', linewidth=2)
        axes[1, col].add_patch(rect_pred_amp)
        axes[1, col].axis('off')
        
        axes[2, col].imshow(y_phase, cmap='gray', vmin=vmin_phase, vmax=vmax_phase)
        rect_gt_phase = plt.Rectangle((center_start, center_start), 
                                       illuminated_size, illuminated_size,
                                       fill=False, edgecolor='red', linewidth=2)
        axes[2, col].add_patch(rect_gt_phase)
        axes[2, col].axis('off')
        
        axes[3, col].imshow(pred_phase, cmap='gray', vmin=vmin_phase, vmax=vmax_phase)
        rect_pred_phase = plt.Rectangle((center_start, center_start), 
                                         illuminated_size, illuminated_size,
                                         fill=False, edgecolor='red', linewidth=2)
        axes[3, col].add_patch(rect_pred_phase)
        axes[3, col].axis('off')
    
    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, fontsize=12, rotation=0, ha='right', va='center')
    
    plt.suptitle('Illuminated Region Comparison (Red Box = Illuminated Area)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.1, 0, 1, 0.96])
    
    save_path = os.path.join(save_dir, 'illuminated_region_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved illuminated region comparison: {save_path}")

def calculate_mse_statistics(overlap_key, test_loader, model, save_dir):
    config = OVERLAP_CONFIGS[overlap_key]
    output_size = config['output_size']
    step = config['step']
    illuminated_size = calculate_illuminated_region_size(step)
    
    mse_amp_list = []
    mse_phase_list = []
    mse_illuminated_amp_list = []
    mse_illuminated_phase_list = []
    mse_outside_amp_list = []
    mse_outside_phase_list = []
    
    center = output_size // 2
    half_illum = illuminated_size // 2
    center_start = center - half_illum
    center_end = center + half_illum
    
    print(f"\n计算 Overlap={overlap_key} 的 MSE 统计...")
    
    for x, y in tqdm(test_loader, desc=f'MSE Calculation (Overlap={overlap_key})'):
        x = x.to(device)
        y_amp = y[:, 0:1, :, :].numpy()
        y_phase = y[:, 1:2, :, :].numpy()
        
        with torch.no_grad():
            ptycho_amp, ptycho_phase = model(x)
        
        ptycho_amp_np = ptycho_amp.cpu().numpy()
        ptycho_phase_np = ptycho_phase.cpu().numpy()
        
        for i in range(x.size(0)):
            mse_amp = np.mean((ptycho_amp_np[i, 0] - y_amp[i, 0]) ** 2)
            mse_phase = np.mean((ptycho_phase_np[i, 0] - y_phase[i, 0]) ** 2)
            
            mse_amp_list.append(mse_amp)
            mse_phase_list.append(mse_phase)
            
            if illuminated_size < output_size:
                y_amp_illum = y_amp[i, 0, center_start:center_end, center_start:center_end]
                pred_amp_illum = ptycho_amp_np[i, 0, center_start:center_end, center_start:center_end]
                y_phase_illum = y_phase[i, 0, center_start:center_end, center_start:center_end]
                pred_phase_illum = ptycho_phase_np[i, 0, center_start:center_end, center_start:center_end]
                
                # 计算照明区域外的区域
                mask = np.ones_like(y_amp[i, 0], dtype=bool)
                mask[center_start:center_end, center_start:center_end] = False
                
                y_amp_outside = y_amp[i, 0][mask]
                pred_amp_outside = ptycho_amp_np[i, 0][mask]
                y_phase_outside = y_phase[i, 0][mask]
                pred_phase_outside = ptycho_phase_np[i, 0][mask]
                
                mse_outside_amp = np.mean((pred_amp_outside - y_amp_outside) ** 2)
                mse_outside_phase = np.mean((pred_phase_outside - y_phase_outside) ** 2)
            else:
                y_amp_illum = y_amp[i, 0]
                pred_amp_illum = ptycho_amp_np[i, 0]
                y_phase_illum = y_phase[i, 0]
                pred_phase_illum = ptycho_phase_np[i, 0]
                
                mse_outside_amp = 0.0
                mse_outside_phase = 0.0
            
            mse_illum_amp = np.mean((pred_amp_illum - y_amp_illum) ** 2)
            mse_illum_phase = np.mean((pred_phase_illum - y_phase_illum) ** 2)
            
            mse_illuminated_amp_list.append(mse_illum_amp)
            mse_illuminated_phase_list.append(mse_illum_phase)
            mse_outside_amp_list.append(mse_outside_amp)
            mse_outside_phase_list.append(mse_outside_phase)
    
    mse_amp_array = np.array(mse_amp_list)
    mse_phase_array = np.array(mse_phase_list)
    mse_illum_amp_array = np.array(mse_illuminated_amp_list)
    mse_illum_phase_array = np.array(mse_illuminated_phase_list)
    mse_outside_amp_array = np.array(mse_outside_amp_list)
    mse_outside_phase_array = np.array(mse_outside_phase_list)
    
    stats = {
        'mse_amp': {
            'mean': float(np.mean(mse_amp_array)),
            'std': float(np.std(mse_amp_array)),
            'min': float(np.min(mse_amp_array)),
            'max': float(np.max(mse_amp_array)),
            'median': float(np.median(mse_amp_array))
        },
        'mse_phase': {
            'mean': float(np.mean(mse_phase_array)),
            'std': float(np.std(mse_phase_array)),
            'min': float(np.min(mse_phase_array)),
            'max': float(np.max(mse_phase_array)),
            'median': float(np.median(mse_phase_array))
        },
        'mse_illuminated_amp': {
            'mean': float(np.mean(mse_illum_amp_array)),
            'std': float(np.std(mse_illum_amp_array)),
            'min': float(np.min(mse_illum_amp_array)),
            'max': float(np.max(mse_illum_amp_array)),
            'median': float(np.median(mse_illum_amp_array))
        },
        'mse_illuminated_phase': {
            'mean': float(np.mean(mse_illum_phase_array)),
            'std': float(np.std(mse_illum_phase_array)),
            'min': float(np.min(mse_illum_phase_array)),
            'max': float(np.max(mse_illum_phase_array)),
            'median': float(np.median(mse_illum_phase_array))
        },
        'mse_outside_amp': {
            'mean': float(np.mean(mse_outside_amp_array)),
            'std': float(np.std(mse_outside_amp_array)),
            'min': float(np.min(mse_outside_amp_array)),
            'max': float(np.max(mse_outside_amp_array)),
            'median': float(np.median(mse_outside_amp_array))
        },
        'mse_outside_phase': {
            'mean': float(np.mean(mse_outside_phase_array)),
            'std': float(np.std(mse_outside_phase_array)),
            'min': float(np.min(mse_outside_phase_array)),
            'max': float(np.max(mse_outside_phase_array)),
            'median': float(np.median(mse_outside_phase_array))
        }
    }
    
    np.savez(
        os.path.join(save_dir, f'mse_data_overlap_{overlap_key}.npz'),
        mse_amp=mse_amp_array,
        mse_phase=mse_phase_array,
        mse_illuminated_amp=mse_illum_amp_array,
        mse_illuminated_phase=mse_illum_phase_array,
        mse_outside_amp=mse_outside_amp_array,
        mse_outside_phase=mse_outside_phase_array
    )
    
    with open(os.path.join(save_dir, f'statistics_overlap_{overlap_key}.txt'), 'w') as f:
        f.write(f"Overlap = {overlap_key}\n")
        f.write(f"Output Size = {output_size}×{output_size}\n")
        f.write(f"Illuminated Size = {illuminated_size}×{illuminated_size}\n")
        f.write(f"Step = {step}\n")
        f.write("="*50 + "\n\n")
        f.write("MSE Amplitude Statistics:\n")
        for key, val in stats['mse_amp'].items():
            f.write(f"  {key}: {val:.6f}\n")
        f.write("\nMSE Phase Statistics:\n")
        for key, val in stats['mse_phase'].items():
            f.write(f"  {key}: {val:.6f}\n")
        f.write("\nMSE Illuminated Amplitude Statistics:\n")
        for key, val in stats['mse_illuminated_amp'].items():
            f.write(f"  {key}: {val:.6f}\n")
        f.write("\nMSE Illuminated Phase Statistics:\n")
        for key, val in stats['mse_illuminated_phase'].items():
            f.write(f"  {key}: {val:.6f}\n")
        f.write("\nMSE Outside Illuminated Amplitude Statistics:\n")
        for key, val in stats['mse_outside_amp'].items():
            f.write(f"  {key}: {val:.6f}\n")
        f.write("\nMSE Outside Illuminated Phase Statistics:\n")
        for key, val in stats['mse_outside_phase'].items():
            f.write(f"  {key}: {val:.6f}\n")
    
    print(f"保存 MSE 数据: mse_data_overlap_{overlap_key}.npz")
    
    return stats, illuminated_size

def generate_comparison_table(all_stats, save_dir):
    overlap_keys = ['0', '0.25', '0.5', '0.7']
    
    table_data = []
    for key in overlap_keys:
        config = OVERLAP_CONFIGS[key]
        illuminated_size = calculate_illuminated_region_size(config['step'])
        row = {
            'Overlap': key,
            'Output Size': f"{config['output_size']}×{config['output_size']}",
            'Step': config['step'],
            'Illum. Size': f"{illuminated_size}×{illuminated_size}",
            'MSE Amp (mean±std)': f"{all_stats[key]['mse_amp']['mean']:.4f}±{all_stats[key]['mse_amp']['std']:.4f}",
            'MSE Phase (mean±std)': f"{all_stats[key]['mse_phase']['mean']:.4f}±{all_stats[key]['mse_phase']['std']:.4f}",
            'MSE Illum Amp': f"{all_stats[key]['mse_illuminated_amp']['mean']:.4f}",
            'MSE Illum Phase': f"{all_stats[key]['mse_illuminated_phase']['mean']:.4f}",
            'MSE Outside Amp': f"{all_stats[key]['mse_outside_amp']['mean']:.4f}",
            'MSE Outside Phase': f"{all_stats[key]['mse_outside_phase']['mean']:.4f}",
        }
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(save_dir, 'comparison_table.csv'), index=False)
    
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.axis('off')
    ax.axis('tight')
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0']*len(df.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    plt.title('TransUNet-Ptycho Performance Comparison Across Different Overlaps', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存对比表格: comparison_table.csv, comparison_table.png")

def generate_mse_distribution_plot(all_mse_data, save_dir):
    overlap_keys = ['0', '0.25', '0.5', '0.7']
    colors = ['#9b59b6', '#2ecc71', '#3498db', '#e74c3c']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for i, (key, color) in enumerate(zip(overlap_keys, colors)):
        ax = axes[i // 2, i % 2]
        mse_amp = all_mse_data[key]['mse_amp']
        mse_illum = all_mse_data[key]['mse_illuminated_amp']
        mse_outside = all_mse_data[key]['mse_outside_amp']
        
        ax.hist(mse_amp, bins=50, alpha=0.4, color=color, label='Full MSE', edgecolor='black')
        ax.hist(mse_illum, bins=50, alpha=0.4, color='orange', label='Illum. MSE', edgecolor='black')
        ax.hist(mse_outside, bins=50, alpha=0.4, color='green', label='Outside MSE', edgecolor='black')
        
        ax.axvline(np.mean(mse_amp), color=color, linestyle='--', linewidth=2, label=f'Full Mean: {np.mean(mse_amp):.4f}')
        ax.axvline(np.mean(mse_illum), color='orange', linestyle='--', linewidth=2, label=f'Illum Mean: {np.mean(mse_illum):.4f}')
        ax.axvline(np.mean(mse_outside), color='green', linestyle='--', linewidth=2, label=f'Outside Mean: {np.mean(mse_outside):.4f}')
        
        ax.set_xlabel('MSE')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Overlap = {key}')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('MSE Distribution Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, 'mse_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存 MSE 分布图: mse_distribution.png")

def main():
    OUTPUT_DIR = os.path.join('d:/BaiduSyncdisk/second_term/Uptynet/article_experiment/overlap_comparison')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'visualizations', 'overlap_0.7'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'statistics'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'illuminated_region'), exist_ok=True)
    
    print("="*60)
    print("Ptycho Overlap Comparison Analysis (Revised)")
    print("="*60)
    
    print("\n" + "="*60)
    print("任务1: 生成对比图（仅 overlap=0.7）")
    print("="*60)
    models = load_models_for_visualization()
    config_07 = OVERLAP_CONFIGS['0.7']
    test_dataset_07 = PtychoDataset(config_07['data_dir'], split='test', output_size=config_07['output_size'])
    generate_comparison_visualization('0.7', test_dataset_07, models, os.path.join(OUTPUT_DIR, 'visualizations', 'overlap_0.7'))
    
    all_stats = {}
    all_mse_data = {}
    test_datasets = {}
    models_dict = {}
    
    print("\n" + "="*60)
    print("任务2 & 3: MSE 统计和照明区域 MSE 计算")
    print("="*60)
    
    for overlap_key in ['0', '0.25', '0.5', '0.7']:
        config = OVERLAP_CONFIGS[overlap_key]
        test_dataset = PtychoDataset(config['data_dir'], split='test', output_size=config['output_size'])
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        test_datasets[overlap_key] = test_dataset
        models_dict[overlap_key] = load_ptycho_model(overlap_key)
        
        stats_dir = os.path.join(OUTPUT_DIR, 'statistics')
        illum_dir = os.path.join(OUTPUT_DIR, 'illuminated_region')
        
        stats, illuminated_size = calculate_mse_statistics(overlap_key, test_loader, models_dict[overlap_key], stats_dir)
        all_stats[overlap_key] = stats
        
        mse_file = np.load(os.path.join(stats_dir, f'mse_data_overlap_{overlap_key}.npz'))
        all_mse_data[overlap_key] = {
            'mse_amp': mse_file['mse_amp'],
            'mse_illuminated_amp': mse_file['mse_illuminated_amp'],
            'mse_outside_amp': mse_file['mse_outside_amp']
        }
        
        print(f"Overlap={overlap_key}: 输出 {config['output_size']}×{config['output_size']}, 照明区域 {illuminated_size}×{illuminated_size}")
    
    print("\n" + "="*60)
    print("任务4: 生成照明区域对比图")
    print("="*60)
    generate_illuminated_region_comparison(test_datasets, models_dict, os.path.join(OUTPUT_DIR, 'illuminated_region'))
    
    print("\n" + "="*60)
    print("生成综合对比表格和分布图")
    print("="*60)
    generate_comparison_table(all_stats, os.path.join(OUTPUT_DIR, 'statistics'))
    generate_mse_distribution_plot(all_mse_data, os.path.join(OUTPUT_DIR, 'statistics'))
    
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)
    print(f"所有结果保存在: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
