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


class PtychoParams:
    def __init__(self):
        self.wavelength = 632.8e-9
        self.angle = 0
        self.dxd = 11e-6
        self.bin = 4
        self.z0 = 1.5e-2
        self.Nd = [64, 64]
        self.number_diff_x = 10
        self.number_diff_y = 10
        self.propagator = "Frau"
        self.dxp = None
        self.dxo = None
        self.probesize = 80e-6
        self.overlap = 0.7
        self.step = None
        self.No = None
        self.Np = None
        self.probe = None
        self.pos = None

    def initialize(self):
        self.Np = self.Nd
        if self.propagator == "Frau":
            self.dxp = self.wavelength * self.z0 / self.Nd[0] / self.dxd / self.bin
        elif self.propagator == "Fres":
            self.dxp = self.dxd * self.bin
        elif self.propagator == "RFD":
            self.pixelsize = 4 * 11e-6
            self.dxp = self.wavelength * self.z0 / self.Nd[0] / self.pixelsize
        
        self.dxo = self.dxp
        self.step = int(self.probesize * (1 - self.overlap) / self.dxp)
        if self.step < 1:
            self.step = 1
        
        self.No = self.Nd[0] + (self.number_diff_x - 1) * self.step + 1
        
        self.create_probe()
        self.generate_scan_positions()

    def create_probe(self):
        probe_size_pixels = int(self.probesize / self.dxp)
        probe = torch.zeros(self.No, self.No)
        center = [probe.shape[0] // 2, probe.shape[1] // 2]
        indices = torch.from_numpy(np.indices(probe.shape)).float()
        distances = torch.sqrt((indices[0] - center[0]) ** 2 + (indices[1] - center[1]) ** 2)
        
        radius = probe_size_pixels // 2
        sigma = radius * 0.2
        
        mask = distances < radius
        probe = torch.where(mask, torch.tensor(1.0), torch.tensor(0.0))
        
        edge_distances = distances - (radius - sigma)
        edge_mask = (distances >= (radius - sigma)) & (distances < radius)
        gaussian_weights = torch.exp(-edge_distances ** 2 / (2 * sigma ** 2))
        probe = torch.where(edge_mask, gaussian_weights, probe)
        
        self.probe = probe.to(device)

    def generate_scan_positions(self):
        X = []
        Y = []
        scan_sep = self.step
        N_scans_x = self.number_diff_x
        N_scans_y = self.number_diff_y
        
        for row in range(N_scans_y):
            for col in range(N_scans_x):
                x_start = col * scan_sep
                y_start = (N_scans_y - row - 1) * scan_sep
                X.append(x_start)
                Y.append(y_start)
        
        self.pos = torch.tensor(list(zip(X, Y)), dtype=torch.float32).to(device)


ptycho_params = PtychoParams()
ptycho_params.initialize()


def process_elements_torch(elems):
    pos, amp, phase = elems
    r = int(pos[1])
    c = int(pos[0])
    o_real = amp[r:r + int(ptycho_params.Nd[0]), c:c + int(ptycho_params.Nd[1])]
    o_im = phase[r:r + int(ptycho_params.Nd[0]), c:c + int(ptycho_params.Nd[1])]
    o_real_reshaped = o_real.view(ptycho_params.Nd[0], ptycho_params.Nd[1])
    o_im_reshaped = o_im.view(ptycho_params.Nd[0], ptycho_params.Nd[1])
    real = o_real_reshaped * torch.cos(np.pi * o_im_reshaped)
    imag = o_real_reshaped * torch.sin(np.pi * o_im_reshaped)
    psi = torch.complex(real, imag)
    return psi


def Frau_create(elem):
    Psi, probe, complex_fourier0, complex_fourier1 = elem
    Psi_ = probe * Psi
    F_ = torch.fft.fftn(Psi_ * complex_fourier1, dim=(0, 1))
    F = complex_fourier0 * torch.fft.fftshift(F_, dim=(0, 1))
    return F


def forward_propagation(amp_out, phase_out, params):
    B = amp_out.shape[0]
    num = params.number_diff_x * params.number_diff_y
    
    pi = torch.tensor(3.1415926535, dtype=torch.cfloat).to(device)
    lambda_ = params.wavelength
    z = torch.tensor(params.z0, dtype=torch.cfloat).to(device)
    dx = params.dxd * params.bin
    L = dx * params.Nd[0]
    k = 2 * pi / lambda_
    
    fx = torch.arange(-1/(2*dx), 1/(2*dx), 1/L).to(device)
    FX, FY = torch.meshgrid([fx, fx], indexing='xy')
    
    x = (dx * torch.arange(-params.Nd[0]//2, params.Nd[0]//2, 1)).to(device)
    dx0 = params.dxo
    L0 = dx0 * params.Nd[0]
    x0 = (dx0 * torch.arange(-params.Nd[0]//2, params.Nd[0]//2, 1)).to(device)
    X, Y = torch.meshgrid([x, x], indexing='xy')
    X0, Y0 = torch.meshgrid([x0, x0], indexing='xy')
    
    delta0 = k / (2 * z) * (X ** 2 + Y ** 2)
    delta1 = k / (2 * z) * (X0 ** 2 + Y0 ** 2)
    complex_fourier0 = dx0 * dx0 * 1 / (1j * lambda_ * z) * (torch.cos(delta0) + 1j * torch.sin(delta0))
    complex_fourier1 = (torch.cos(delta1) + 1j * torch.sin(delta1))
    
    probe = params.probe
    sub_arr_size = params.Nd[0]
    center_i = params.No // 2
    center_j = params.No // 2
    start_i = center_i - sub_arr_size // 2
    end_i = start_i + sub_arr_size
    start_j = center_j - sub_arr_size // 2
    end_j = start_j + sub_arr_size
    probe_sub = probe[start_i:end_i, start_j:end_j]
    
    all_diffractions = []
    
    for b in range(B):
        amp = amp_out[b, 0]
        phase = phase_out[b, 0]
        
        amp_expanded = amp.repeat(num, 1, 1)
        phase_expanded = phase.repeat(num, 1, 1)
        
        elems = (params.pos, amp_expanded, phase_expanded)
        Psi = list(map(process_elements_torch, zip(*elems)))
        Psi = torch.stack(Psi, dim=0)
        
        probe_ex = probe_sub.unsqueeze(0).repeat(num, 1, 1)
        complex_fourier0_ex = complex_fourier0.unsqueeze(0).repeat(num, 1, 1)
        complex_fourier1_ex = complex_fourier1.unsqueeze(0).repeat(num, 1, 1)
        
        elem = (Psi, probe_ex, complex_fourier0_ex, complex_fourier1_ex)
        Psi2 = torch.stack([Frau_create(e) for e in zip(*elem)])
        
        diffraction_patterns = torch.abs(Psi2) ** 2
        
        max_vals = diffraction_patterns.view(num, -1).max(dim=1)[0].view(num, 1, 1)
        max_vals = torch.clamp(max_vals, min=1e-8)
        diffraction_patterns = diffraction_patterns / max_vals
        
        all_diffractions.append(diffraction_patterns)
    
    return torch.stack(all_diffractions, dim=0)


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


class TransUNetPtychoPadded(nn.Module):
    def __init__(self, img_size=128, in_channels=100, embed_dim=256, depth=2, num_heads=8, mlp_ratio=4.0):
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
        self.outc_amp = nn.Conv2d(64, 1, 3, padding='same')
        
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
        self.outc_phase = nn.Conv2d(64, 1, 3, padding='same')
    
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


class PtychoDatasetPadded(Dataset):
    """加载 padded 数据 (作为模型输入) 与 unpadded 数据 (作为前向传播损失参考)"""
    def __init__(self, data_root, split='train', target_multiple_of=8, original_size=64):
        self.split = split
        self.input_dir = os.path.join(data_root, split, 'inputs')
        self.label_dir = os.path.join(data_root, split, 'labels')
        self.input_files = sorted(glob.glob(os.path.join(self.input_dir, 'input_*.npy')))
        self.target_multiple_of = target_multiple_of
        self.original_size = original_size

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = self.input_files[idx]
        base_name = os.path.basename(input_path).replace('input_', '')
        input_array = np.load(input_path).astype(np.float32)
        H, W = input_array.shape[1], input_array.shape[2]
        # padding 到能被 target_multiple_of 整除的尺寸
        new_H = ((H + self.target_multiple_of - 1) // self.target_multiple_of) * self.target_multiple_of
        new_W = ((W + self.target_multiple_of - 1) // self.target_multiple_of) * self.target_multiple_of
        pad_h = new_H - H
        pad_w = new_W - W
        if pad_h > 0 or pad_w > 0:
            input_array = np.pad(input_array, ((0, 0), (0, pad_h), (0, pad_w)),
                                  mode='constant', constant_values=0.0)
        input_tensor = torch.from_numpy(input_array)

        label_path = os.path.join(self.label_dir, f'label_{base_name}')
        label_array = np.load(label_path).astype(np.float32)
        LH, LW = label_array.shape[1], label_array.shape[2]
        pad_lh = new_H - LH
        pad_lw = new_W - LW
        if pad_lh > 0 or pad_lw > 0:
            label_array = np.pad(label_array, ((0, 0), (0, pad_lh), (0, pad_lw)),
                                  mode='constant', constant_values=0.0)
        label_tensor = torch.from_numpy(label_array)

        # 从 padded 数据中提取中心 original_size×original_size 区域作为 unpadded 参考
        padded_H, padded_W = input_array.shape[1], input_array.shape[2]
        start_h = (padded_H - self.original_size) // 2
        start_w = (padded_W - self.original_size) // 2
        unpadded_array = input_array[:, start_h:start_h+self.original_size, start_w:start_w+self.original_size]
        unpadded_tensor = torch.from_numpy(unpadded_array)

        return input_tensor, label_tensor, unpadded_tensor


def train_ptycho_model(model, model_name, train_loader, criterion, optimizer, scheduler, epochs=80, device='cuda', lambda_forward=0.1, model_save_dir=None, train_results_dir=None):
    if model_save_dir is None:
        model_save_dir = f'{BASE_DIR}/models_chip_padded'
    if train_results_dir is None:
        train_results_dir = f'{BASE_DIR}/train_results_chip_padded'
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(train_results_dir, exist_ok=True)
    print(f"\n训练 {model_name} (带前向传播损失)...")
    print("=" * 80)
    print(f"  模型保存目录: {model_save_dir}")
    print(f"  训练结果目录: {train_results_dir}")
    train_losses = []
    train_supervised_losses = []
    train_forward_losses = []
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_supervised_loss = 0.0
        total_forward_loss = 0.0
        for batch_idx, (x, y, x_unpadded) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')):
            x = x.to(device)
            y_i = y[:, 0:1, :, :].to(device)
            y_phi = y[:, 1:2, :, :].to(device)
            x_unpadded = x_unpadded.to(device)
            optimizer.zero_grad()
            
            pred_i, pred_phi = model(x)
            
            loss_supervised_i = criterion(pred_i, y_i)
            loss_supervised_phi = criterion(pred_phi, y_phi)
            loss_supervised = loss_supervised_i + loss_supervised_phi
            
            # 前向传播: 从预测的振幅/相位反演衍射图案, 与 unpadded 衍射图比较
            pred_diffraction = forward_propagation(pred_i, pred_phi, ptycho_params)
            # 将参考和预测都转换为振幅域进行比较
            ref = torch.sqrt(torch.clamp(x_unpadded, min=0.0))
            pred_amp = torch.sqrt(torch.clamp(pred_diffraction, min=0.0))
            # 计算 L2 损失
            loss_forward = torch.norm(pred_amp - ref, p=2)
            
            loss = loss_supervised + lambda_forward * loss_forward
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total_supervised_loss += loss_supervised.item() * x.size(0)
            total_forward_loss += loss_forward.item() * x.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        avg_supervised_loss = total_supervised_loss / len(train_loader.dataset)
        avg_forward_loss = total_forward_loss / len(train_loader.dataset)
        
        train_losses.append(avg_loss)
        train_supervised_losses.append(avg_supervised_loss)
        train_forward_losses.append(avg_forward_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'{model_save_dir}/best_{model_name.lower()}_ptycho.pth')
        
        scheduler.step(avg_loss)
        
        print(f"{model_name} - Epoch [{epoch+1:3d}/{epochs}]")
        print(f"  Total Loss: {avg_loss:.6f}")
        print(f"  Supervised Loss: {avg_supervised_loss:.6f}")
        print(f"  Forward Loss: {avg_forward_loss:.6f}")
    
    torch.save(model.state_dict(), f'{model_save_dir}/{model_name.lower()}_ptycho_weights.pth')
    print(f"\n{model_name} 训练完成！")
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs+1), train_losses, 'g-', linewidth=2, label='Total Loss')
    plt.plot(range(1, epochs+1), train_supervised_losses, 'b--', linewidth=2, label='Supervised Loss')
    plt.plot(range(1, epochs+1), [lambda_forward * l for l in train_forward_losses], 'r--', linewidth=2, label=f'Forward Loss (×{lambda_forward})')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training Loss (Padded with Forward Loss, overlap {ptycho_params.overlap})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{train_results_dir}/loss_padded_forward.png', dpi=300, bbox_inches='tight')
    
    return train_losses


def test_ptycho_model(model, test_loader, device='cuda'):
    model.eval()
    all_pred_I = []
    all_pred_phi = []
    all_Y_I = []
    all_Y_phi = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试中"):
            # data is (x, y, x_unpadded)
            x = batch[0]
            y = batch[1]
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


if __name__ == '__main__':
    DATASET_DIR = f'{BASE_DIR}/data_ptycho_chip_padded'
    UNPADDED_DIR = f'{BASE_DIR}/data_ptycho_chip_padded'
    MODEL_DIR = f'{BASE_DIR}/models_chip_padded'
    TRAIN_RESULTS_DIR = f'{BASE_DIR}/train_results_chip_padded_overlap_0.7'
    TEST_RESULTS_DIR = f'{BASE_DIR}/test_results_chip_padded_overlap_0.7'
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(TRAIN_RESULTS_DIR, exist_ok=True)
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    
    print("加载芯片缺陷数据集（overlap 0.7）...")
    train_dataset = PtychoDatasetPadded(DATASET_DIR, split='train')
    test_dataset = PtychoDatasetPadded(DATASET_DIR, split='test')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    print(f"训练样本数: {len(train_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")
    
    x, y, xu = train_dataset[0]
    print(f"输入尺寸: {x.shape}")
    print(f"标签尺寸: {y.shape}")
    print(f"Unpadded 尺寸: {xu.shape}")
    
    num_channels = x.shape[0]
    input_size = x.shape[1]
    output_size = y.shape[2]
    print(f"检测到输入通道数: {num_channels}")
    print(f"检测到输入尺寸: {input_size}×{input_size}")
    print(f"检测到输出尺寸: {output_size}×{output_size}")
    
    EPOCHS = 5
    LR = 1e-4
    LAMBDA_FORWARD = 0.01
    
    ptycho_params.overlap = 0.7
    ptycho_params.No = output_size
    ptycho_params.initialize()
    
    print(f"\n叠层成像参数:")
    print(f"  波长: {ptycho_params.wavelength * 1e9:.1f} nm")
    print(f"  探针大小: {ptycho_params.probesize * 1e6:.1f} μm")
    print(f"  重叠率: {ptycho_params.overlap * 100:.1f}%")
    print(f"  扫描步距: {ptycho_params.step} 像素")
    print(f"  扫描网格: {ptycho_params.number_diff_x}×{ptycho_params.number_diff_y}")
    print(f"  衍射图数量: {ptycho_params.number_diff_x * ptycho_params.number_diff_y}")
    print(f"  对象尺寸 (No): {ptycho_params.No}×{ptycho_params.No}")
    
    model = TransUNetPtychoPadded(
        img_size=input_size,
        in_channels=num_channels,
        embed_dim=256,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0
    ).to(device)
    
    pretrained_path = f'{MODEL_DIR}/best_transunet-ptycho-padded_ptycho.pth'
    if os.path.exists(pretrained_path):
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        if isinstance(pretrained_dict, dict) and 'model_state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['model_state_dict']
        model_dict = model.state_dict()
        filtered = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        print(f"  已加载预训练权重: {pretrained_path}")
    else:
        print(f"  警告: 未找到预训练权重 {pretrained_path}，使用随机初始化")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-6)
    
    print(f"\n训练配置:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LR}")
    print(f"  Lambda Forward: {LAMBDA_FORWARD}")
    print(f"  数据集: overlap 0.7 ({DATASET_DIR})")
    print(f"  初始化权重: 预训练权重 ({pretrained_path})")
    
    losses = train_ptycho_model(model, "TransUNet-Ptycho-Padded", train_loader, criterion, optimizer, scheduler, EPOCHS, device, lambda_forward=LAMBDA_FORWARD, model_save_dir=MODEL_DIR, train_results_dir=TRAIN_RESULTS_DIR)
    
    pred_I, pred_phi, Y_I, Y_phi, mse_I, mse_phi, mae_I, mae_phi = test_ptycho_model(model, test_loader)
    
    print("\n" + "=" * 80)
    print("=== TransUNet-Ptycho-Padded 测试集评估结果 (overlap 0.7) ===")
    print("=" * 80)
    print(f"振幅 MSE: {mse_I:.6f}, 相位 MSE: {mse_phi:.6f}")
    print(f"振幅 MAE: {mae_I:.6f}, 相位 MAE: {mae_phi:.6f}")
    print("=" * 80)
    
    np.savez(f'{TEST_RESULTS_DIR}/ptycho_predictions_padded.npz',
             pred_I=pred_I, pred_phi=pred_phi, Y_I=Y_I, Y_phi=Y_phi)
    
    with open(f'{TEST_RESULTS_DIR}/ptycho_evaluation_padded.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TransUNet-Ptycho-Padded 测试结果 (overlap 0.7)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"输入通道数: {num_channels}\n")
        f.write(f"输入尺寸: {input_size}×{input_size}\n")
        f.write(f"输出尺寸: {output_size}×{output_size}\n\n")
        f.write(f"振幅 MSE: {mse_I:.6f}\n")
        f.write(f"相位 MSE: {mse_phi:.6f}\n")
        f.write(f"振幅 MAE: {mae_I:.6f}\n")
        f.write(f"相位 MAE: {mae_phi:.6f}\n")
    
    # 计算照明区域 (按原始 64 尺寸标记照明)
    raw_size = 64
    center_start = (output_size - raw_size) // 2
    center_end = center_start + raw_size
    
    print("\n保存测试集对比图片（带照明区域标记）...")
    num_samples_to_save = 100
    for i in range(min(num_samples_to_save, len(pred_I))):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        ax0 = axes[0, 0]
        im0 = ax0.imshow(Y_I[i, 0], cmap='gray', vmin=0, vmax=1)
        rect0 = plt.Rectangle((center_start, center_start), raw_size, raw_size, 
                             fill=False, edgecolor='red', linewidth=3)
        ax0.add_patch(rect0)
        ax0.set_title('Ground Truth (Amplitude)', fontsize=14)
        ax0.axis('off')
        
        ax1 = axes[0, 1]
        im1 = ax1.imshow(pred_I[i, 0], cmap='gray', vmin=0, vmax=1)
        rect1 = plt.Rectangle((center_start, center_start), raw_size, raw_size, 
                             fill=False, edgecolor='red', linewidth=3)
        ax1.add_patch(rect1)
        ax1.set_title('TransUNet-Ptycho-Padded (Amplitude)', fontsize=14)
        ax1.axis('off')
        
        axes[0, 2].imshow(np.abs(pred_I[i, 0] - Y_I[i, 0]), cmap='jet', vmin=0, vmax=0.2)
        axes[0, 2].set_title('Amplitude Error', fontsize=14)
        axes[0, 2].axis('off')
        
        ax3 = axes[1, 0]
        im3 = ax3.imshow(Y_phi[i, 0], cmap='gray', vmin=0, vmax=1)
        rect3 = plt.Rectangle((center_start, center_start), raw_size, raw_size, 
                             fill=False, edgecolor='red', linewidth=3)
        ax3.add_patch(rect3)
        ax3.set_title('Ground Truth (Phase)', fontsize=14)
        ax3.axis('off')
        
        ax4 = axes[1, 1]
        im4 = ax4.imshow(pred_phi[i, 0], cmap='gray', vmin=0, vmax=1)
        rect4 = plt.Rectangle((center_start, center_start), raw_size, raw_size, 
                             fill=False, edgecolor='red', linewidth=3)
        ax4.add_patch(rect4)
        ax4.set_title('TransUNet-Ptycho-Padded (Phase)', fontsize=14)
        ax4.axis('off')
        
        axes[1, 2].imshow(np.abs(pred_phi[i, 0] - Y_phi[i, 0]), cmap='jet', vmin=0, vmax=0.2)
        axes[1, 2].set_title('Phase Error', fontsize=14)
        axes[1, 2].axis('off')
        
        plt.suptitle(f'TransUNet-Ptycho-Padded (overlap 0.7) Test Sample {i+1}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{TEST_RESULTS_DIR}/ptycho_sample_{i+1}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"已保存 {min(num_samples_to_save, len(pred_I))} 组测试集对比图片（红色方框标记照明区域）")
    print("\nTransUNet-Ptycho-Padded 芯片数据集训练和测试完成（overlap 0.7）!")
