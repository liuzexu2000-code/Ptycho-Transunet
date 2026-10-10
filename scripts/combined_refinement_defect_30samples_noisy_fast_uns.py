"""
联合叠层成像细化与缺陷检测脚本 —— 30个样本，带噪声版本（快速无监督优化）
=============================================================================
功能：在输入衍射图中添加泊松噪声 + 高斯噪声后，再进行 refinement 对比
1) 加载预训练 TransUNet 得到初始重构
2) 使用传统 mPIE 细化（基于振幅替换迭代）
3) 使用基于梯度的无监督细化（快速版本：向量化前向传播）
4) 生成对比图（红色方框照明区域内裁剪显示）
5) 对 mPIE 与无监督细化结果分别进行缺陷检测

噪声模型：
- 泊松噪声：模拟光子计数噪声，与信号强度相关
- 高斯噪声：模拟电子噪声/暗电流，独立于信号强度
输出：
    combined_refinement_defect_30samples_noisy_fast_uns/
      ├── sample_XXX/
      │     ├── comparison_all_methods.png
      │     ├── comparison_with_init.png
      │     ├── comparison_random_init.png
      │     ├── comparison_diff_maps.png
      │     ├── defect_detection_all.png
      │     ├── comparison_phase_all.png
      │     ├── unsupervised_loss.png
      │     ├── *_amp.npy / *_phase.npy
      │     └── mse_roi.txt
      ├── summary_mse.txt
      ├── mse_boxplot_vs_defect.png
      ├── mse_boxplot_vs_clean.png
      └── mse_per_sample_vs_defect.png
"""

import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# --------------------------------------------------------------------------
# 全局路径与设备
# --------------------------------------------------------------------------
BASE_DIR = r"d:/BaiduSyncdisk/second_term/Uptynet/article_experiment"
DATA_DIR = os.path.join(BASE_DIR, "data_ptycho_chip_padded", "test", "inputs")
LABEL_DIR = os.path.join(BASE_DIR, "data_ptycho_chip_padded", "test", "labels")
MODEL_PATH = os.path.join(BASE_DIR, "models_chip_padded_noisy", "best_transunet-ptycho-padded_ptycho.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "combined_refinement_defect_30samples_noisy")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{DEVICE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

EPS_PIE = 1e-8

# --------------------------------------------------------------------------
# PIE 算法配置类
# --------------------------------------------------------------------------
class PIEConfig:
    def __init__(
        self,
        algorithm: str = "epie",
        iterations: int = 30,
        beta_object: float = 0.8,
        alpha_object: float = 0.1,
        momentum_friction: float = 0.7,
        momentum_feedback: float = 0.3,
        momentum_start: int = 5,
        random_order: bool = True,
        seed: int = 0,
        eps: float = 1e-8,
        progress_every: int = 5,
    ):
        self.algorithm = algorithm
        self.iterations = iterations
        self.beta_object = beta_object
        self.alpha_object = alpha_object
        self.momentum_friction = momentum_friction
        self.momentum_feedback = momentum_feedback
        self.momentum_start = momentum_start
        self.random_order = random_order
        self.seed = seed
        self.eps = eps
        self.progress_every = progress_every


def _pie_denominator(probe_intensity, algorithm, alpha, eps):
    max_val = probe_intensity.max().clamp_min(eps)
    if algorithm == "epie":
        return max_val
    return (1.0 - alpha) * probe_intensity + alpha * max_val + eps


@torch.no_grad()
def run_pie_iteration(
    obj_init: torch.Tensor,
    probe_fixed: torch.Tensor,
    positions: torch.Tensor,
    measured_amp: torch.Tensor,
    cf0: torch.Tensor,
    cf1: torch.Tensor,
    config: PIEConfig,
    device=DEVICE,
):
    algorithm = config.algorithm.lower()
    assert algorithm in {"epie", "rpie", "mpie"}, f"未知 algorithm: {config.algorithm}"

    obj = obj_init.clone().to(torch.complex64).to(device)
    probe_c = probe_fixed.to(torch.complex64).to(device)
    probe_intensity = probe_c.abs().square()

    n_frames = positions.shape[0]
    Nd = probe_c.shape[0]
    No = obj.shape[0]

    errors = []
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    obj_momentum = torch.zeros_like(obj)
    obj_buffer = obj.clone()

    for it in range(config.iterations):
        if config.random_order:
            order = torch.randperm(n_frames, generator=generator).tolist()
        else:
            order = list(range(n_frames))

        amp_error = torch.zeros((), dtype=torch.float64, device=device)
        amp_energy = torch.zeros((), dtype=torch.float64, device=device)

        for fi in order:
            r, c = positions[fi].tolist()
            object_patch = obj[r:r + Nd, c:c + Nd].clone()

            exit_wave = object_patch * probe_c
            detector_wave = torch.fft.fftshift(
                torch.fft.fft2(exit_wave * cf1, norm="ortho"), dim=(-2, -1)
            ) * cf0
            predicted_amp = detector_wave.abs()
            target_amp = measured_amp[fi].to(device).to(torch.float32)

            new_wave = target_amp.to(torch.complex64) * detector_wave / (predicted_amp + config.eps)

            updated_exit = torch.fft.ifft2(
                torch.fft.ifftshift(new_wave / cf0, dim=(-2, -1)), norm="ortho"
            ) / cf1

            delta = updated_exit - exit_wave

            denom_obj = _pie_denominator(probe_intensity, algorithm, config.alpha_object, config.eps)

            obj_patch_new = object_patch + config.beta_object * torch.conj(probe_c) * delta / denom_obj
            obj[r:r + Nd, c:c + Nd] = obj_patch_new

            amp_error += torch.sum((predicted_amp.double() - target_amp.double()).square())
            amp_energy += torch.sum(target_amp.double().square())

        if algorithm == "mpie" and it >= config.momentum_start:
            grad = obj_buffer - obj
            obj_momentum = grad + config.momentum_friction * obj_momentum
            obj = obj - config.momentum_feedback * obj_momentum
            obj_buffer = obj.clone()

        err = torch.sqrt(amp_error / amp_energy.clamp_min(config.eps)).item()
        errors.append(err)

    obj_np = obj.detach().cpu().numpy()
    obj_amp = np.abs(obj_np)
    obj_phase = np.angle(obj_np) / np.pi
    return obj_amp, obj_phase, errors


def create_random_initial_guess(No, seed=0):
    rng = np.random.RandomState(seed)
    amp_np = rng.uniform(0.5, 1.0, size=(No, No))
    phase_np = rng.uniform(-1.0, 1.0, size=(No, No))
    obj_c = torch.complex(
        torch.from_numpy(amp_np * np.cos(np.pi * phase_np)).float(),
        torch.from_numpy(amp_np * np.sin(np.pi * phase_np)).float(),
    ).to(DEVICE)
    return amp_np, phase_np, obj_c


# --------------------------------------------------------------------------
# 噪声参数配置
# --------------------------------------------------------------------------
NOISE_CONFIG = {
    'add_poisson': True,        # 是否添加泊松噪声
    'add_gaussian': True,       # 是否添加高斯噪声
    'poisson_scale': 1.0,       # 泊松噪声缩放因子，越小噪声越大
    'gaussian_snr_db': 30,      # 高斯噪声目标 SNR (dB)，越大噪声越小
    'seed': 2024,               # 随机种子
}

print(f"噪声配置:")
print(f"  添加泊松噪声: {NOISE_CONFIG['add_poisson']}")
print(f"  添加高斯噪声: {NOISE_CONFIG['add_gaussian']}")
if NOISE_CONFIG['add_poisson']:
    print(f"  泊松噪声缩放因子: {NOISE_CONFIG['poisson_scale']}")
if NOISE_CONFIG['add_gaussian']:
    print(f"  高斯噪声目标 SNR: {NOISE_CONFIG['gaussian_snr_db']} dB")

# --------------------------------------------------------------------------
# 样本选择配置
# --------------------------------------------------------------------------
SAMPLE_CONFIG = {
    'num_samples': 20,           # 测试样本数量
    'random_select': False,       # 是否随机抽取（True=随机，False=按顺序）
    'seed': 2024,                # 随机种子（仅在 random_select=True 时生效）
}

print(f"\n样本配置:")
print(f"  测试样本数量: {SAMPLE_CONFIG['num_samples']}")
print(f"  随机抽取: {SAMPLE_CONFIG['random_select']}")
if SAMPLE_CONFIG['random_select']:
    print(f"  随机种子: {SAMPLE_CONFIG['seed']}")


def add_poisson_noise(image, scale=1.0):
    image_clamped = np.clip(image * scale, 0.0, None)
    noisy = np.random.poisson(image_clamped * 1000.0) / 1000.0
    noisy = noisy / scale
    noisy = np.clip(noisy, 0.0, None)
    return noisy


def add_gaussian_noise(image, target_snr_db):
    image = np.array(image, dtype=np.float64)
    
    signal_power = np.mean(image ** 2)
    if signal_power < 1e-10:
        signal_power = 1e-10
    
    target_snr = 10.0 ** (target_snr_db / 10.0)
    sigma = np.sqrt(signal_power / target_snr)
    
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0.0, None)
    
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-20:
        noise_power = 1e-20
    actual_snr_db = 10.0 * np.log10(signal_power / noise_power)
    
    return noisy_image, actual_snr_db


def add_noise_to_diffraction(input_data, config):
    np.random.seed(config['seed'])
    
    noisy_data = np.copy(input_data)
    noise_info = {}
    
    if config.get('add_poisson', False):
        for i in range(input_data.shape[0]):
            noisy_data[i] = add_poisson_noise(
                input_data[i], 
                scale=config.get('poisson_scale', 1.0)
            )
    
    if config.get('add_gaussian', False):
        noisy_data, gaussian_snr_db = add_gaussian_noise(
            noisy_data, 
            target_snr_db=config['gaussian_snr_db']
        )
        noise_info['gaussian_snr_db'] = float(gaussian_snr_db)
    
    return noisy_data, noise_info


# --------------------------------------------------------------------------
# 1. 参数类 (与原脚本完全一致)
# --------------------------------------------------------------------------
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
        self.pixelsize = None

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
        self.step = int(self.step)
        self.No = self.Nd[0] + (self.number_diff_x - 1) * self.step + 1
        self.create_probe()
        self.generate_scan_positions()

    def create_probe(self):
        probe_size_pixels = int(self.probesize / self.dxp)
        probe = torch.zeros(self.No, self.No)
        indices = torch.from_numpy(np.indices(probe.shape)).float()
        center = [probe.shape[0] // 2, probe.shape[1] // 2]
        distances = torch.sqrt((indices[0] - center[0]) ** 2 + (indices[1] - center[1]) ** 2)
        radius = probe_size_pixels // 2
        sigma = radius * 0.2
        mask = distances < radius
        probe = torch.where(mask, torch.tensor(1.0), torch.tensor(0.0))
        edge_distances = distances - (radius - sigma)
        edge_mask = (distances >= (radius - sigma)) & (distances < radius)
        gaussian_weights = torch.exp(-edge_distances ** 2 / (2 * sigma ** 2))
        probe = torch.where(edge_mask, gaussian_weights, probe)
        self.probe = probe.to(DEVICE)

    def generate_scan_positions(self):
        X, Y = [], []
        scan_sep = self.step
        Nx = self.number_diff_x
        Ny = self.number_diff_y
        for row in range(Ny):
            for col in range(Nx):
                x_start = col * scan_sep
                y_start = (Ny - row - 1) * scan_sep
                X.append(x_start)
                Y.append(y_start)
        self.pos = torch.tensor(list(zip(X, Y)), dtype=torch.float32).to(DEVICE)

import pdb
def compute_illumination_region(params):
    probe_size_pixels = int(params.probesize / params.dxp)
    illumination_width = params.step * (params.number_diff_x - 1) + probe_size_pixels
    illumination_height = params.step * (params.number_diff_y - 1) + probe_size_pixels
    x_min = (params.No - illumination_width) // 2
    x_max = x_min + illumination_width
    y_min = (params.No - illumination_height) // 2
    y_max = y_min + illumination_height
    pdb.set_trace()
    return (x_min, x_max, y_min, y_max)


# --------------------------------------------------------------------------
# 2. TransUNet (与原脚本完全一致)
# --------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding="same")
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
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
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)

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
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransUNetPtychoPadded(nn.Module):
    def __init__(self, img_size=128, in_channels=100, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4.0):
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
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=0.05, attn_drop=0.025)
            for _ in range(depth)
        ])
        self.dec_amp_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResidualBlock(256, 256)
        )
        self.dec_amp_fuse1 = ResidualBlock(256 + 128, 128)
        self.dec_amp_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResidualBlock(128, 128)
        )
        self.dec_amp_fuse2 = ResidualBlock(128 + 64, 64)
        self.dec_amp_up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResidualBlock(64, 64)
        )
        self.outc_amp = nn.Conv2d(64, 1, 3, padding="same")

        self.dec_phase_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResidualBlock(256, 256)
        )
        self.dec_phase_fuse1 = ResidualBlock(256 + 128, 128)
        self.dec_phase_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResidualBlock(128, 128)
        )
        self.dec_phase_fuse2 = ResidualBlock(128 + 64, 64)
        self.dec_phase_up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ResidualBlock(64, 64)
        )
        self.outc_phase = nn.Conv2d(64, 1, 3, padding="same")

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


def load_pretrained_weights(model, pretrained_path):
    if os.path.exists(pretrained_path):
        print(f"加载预训练参数: {pretrained_path}")
        pretrained_dict = torch.load(pretrained_path, map_location=DEVICE)
        if isinstance(pretrained_dict, dict) and 'model_state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['model_state_dict']
        model_dict = model.state_dict()
        filtered = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if len(filtered) < len(model_dict):
            print(f"  警告: 仅成功加载 {len(filtered)}/{len(model_dict)} 层权重")
        else:
            print(f"  成功加载全部 {len(filtered)} 层权重")
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
    else:
        print(f"警告：未找到预训练文件 {pretrained_path}，使用随机初始化")
    return model


# --------------------------------------------------------------------------
# 3. 传统 (mPIE) 振幅替换迭代 —— 探针固定
# --------------------------------------------------------------------------
def build_propagators(params, device=DEVICE):
    lambda_ = params.wavelength
    z = torch.tensor(params.z0, dtype=torch.complex128, device=device)
    dx0 = params.dxo
    Nd = params.Nd[0]
    x0 = (dx0 * torch.arange(-Nd // 2, Nd // 2, dtype=torch.float32, device=device))
    X0, Y0 = torch.meshgrid(x0, x0, indexing='ij')
    pi = torch.tensor(np.pi, dtype=torch.complex128, device=device)
    k = 2 * pi / lambda_
    delta1 = (k / (2 * z)) * (X0 ** 2 + Y0 ** 2)
    cf1 = torch.exp(1j * delta1).to(torch.complex64)

    dx = params.dxd * params.bin
    x_phys = (dx * torch.arange(-Nd // 2, Nd // 2, dtype=torch.float32, device=device))
    Xp, Yp = torch.meshgrid(x_phys, x_phys, indexing='ij')
    delta0 = (k / (2 * z)) * (Xp ** 2 + Yp ** 2)
    pre = (dx0 * dx0) / (1j * lambda_ * z.item())
    cf0 = pre * torch.exp(1j * delta0).to(torch.complex64)
    cf0 = cf0 / (cf0.abs() + EPS_PIE)
    return cf0, cf1


def prepare_positions_for_pie(params, device=DEVICE):
    pos = params.pos.detach().cpu().numpy()
    rc = np.stack([pos[:, 1], pos[:, 0]], axis=-1).astype(np.int64)
    return torch.from_numpy(rc).to(device)


@torch.no_grad()
def run_mpie_iteration(obj_init, probe_small, positions_rc, measured_amp, cf0, cf1, iterations=30, beta=0.8, alpha=0.1, momentum_friction=0.7, momentum_feedback=0.3, momentum_start=5):
    n_frames = positions_rc.shape[0]
    Nd = probe_small.shape[0]
    No = obj_init.shape[0]
    probe_intensity = probe_small.to(torch.float32).to(DEVICE)
    probe_c = probe_small.to(torch.complex64).to(DEVICE)
    obj = obj_init.clone().to(torch.complex64).to(DEVICE)

    errors = []
    generator = torch.Generator(device="cpu").manual_seed(0)
    obj_momentum = torch.zeros_like(obj)
    obj_buffer = obj.clone()

    for it in range(iterations):
        order = torch.randperm(n_frames, generator=generator).tolist()
        amp_error = 0.0
        amp_energy = 0.0

        for fi in order:
            r, c = positions_rc[fi].tolist()
            object_patch = obj[r:r + Nd, c:c + Nd].clone()
            exit_wave = object_patch * probe_c
            detector_wave = torch.fft.fftshift(
                torch.fft.fftn(exit_wave * cf1, dim=(0, 1)),
                dim=(0, 1)
            ) * cf0
            predicted_amp = detector_wave.abs()
            target_amp = measured_amp[fi].to(torch.float32).to(DEVICE)
            new_wave = target_amp.to(torch.complex64) * detector_wave / (predicted_amp + EPS_PIE)
            updated_exit = torch.fft.ifftn(
                torch.fft.ifftshift(new_wave / cf0, dim=(0, 1)),
                dim=(0, 1)
            ) / cf1
            delta = updated_exit - exit_wave
            max_p = probe_intensity.max().clamp_min(EPS_PIE)
            denom = (1.0 - alpha) * probe_intensity + alpha * max_p + EPS_PIE
            obj_patch_new = object_patch + beta * torch.conj(probe_c) * delta / denom
            obj[r:r + Nd, c:c + Nd] = obj_patch_new
            amp_error += torch.sum((predicted_amp.double() - target_amp.double()).square()).item()
            amp_energy += torch.sum(target_amp.double().square()).item()

        if it >= momentum_start:
            grad = obj_buffer - obj
            obj_momentum = grad + momentum_friction * obj_momentum
            obj = obj - momentum_feedback * obj_momentum

        err = float(np.sqrt(max(amp_error, 0.0) / max(amp_energy, EPS_PIE)))
        errors.append(err)

    obj_np = obj.detach().cpu().numpy()
    obj_amp = np.abs(obj_np)
    obj_phase = np.angle(obj_np) / np.pi
    return obj_amp, obj_phase, errors


# --------------------------------------------------------------------------
# 4. 无监督细化（快速版本：向量化前向传播）
# --------------------------------------------------------------------------
def total_variation_loss(image):
    tv_h = torch.pow(image[1:, :] - image[:-1, :], 2).mean()
    tv_w = torch.pow(image[:, 1:] - image[:, :-1], 2).mean()
    return tv_h + tv_w


def high_freq_penalty(x):
    blur = F.avg_pool2d(x, 3, stride=1, padding=1)
    return F.mse_loss(x, blur)


def build_unsupervised_forward_cache(params):
    num = params.number_diff_x * params.number_diff_y
    nd = params.Nd[0]

    pos = params.pos.to(torch.long)
    rows = pos[:, 1]
    cols = pos[:, 0]
    grid = torch.arange(nd, dtype=torch.long, device=DEVICE)
    patch_rows = rows[:, None, None] + grid[None, :, None]
    patch_cols = cols[:, None, None] + grid[None, None, :]

    pi = torch.tensor(np.pi, dtype=torch.complex64, device=DEVICE)
    lambda_ = params.wavelength
    z = torch.tensor(params.z0, dtype=torch.complex64, device=DEVICE)
    dx = params.dxd * params.bin
    dx0 = params.dxo
    k = 2 * pi / lambda_

    x = dx * torch.arange(-nd // 2, nd // 2, 1, dtype=torch.float32, device=DEVICE)
    x0 = dx0 * torch.arange(-nd // 2, nd // 2, 1, dtype=torch.float32, device=DEVICE)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    X0, Y0 = torch.meshgrid(x0, x0, indexing="ij")
    delta0 = k / (2 * z) * (X ** 2 + Y ** 2)
    delta1 = k / (2 * z) * (X0 ** 2 + Y0 ** 2)
    cf0 = dx0 ** 2 / (1j * lambda_ * z) * torch.exp(1j * delta0)
    cf1 = torch.exp(1j * delta1)

    start = (params.No - nd) // 2
    probe_sub = params.probe[start:start + nd, start:start + nd].to(torch.float32)

    return {
        "num": num,
        "nd": nd,
        "patch_rows": patch_rows,
        "patch_cols": patch_cols,
        "probe_sub": probe_sub,
        "cf0": cf0.to(torch.complex64),
        "cf1": cf1.to(torch.complex64),
    }


def forward_propagation_fast(amp_out, phase_out, cache):
    all_diffractions = []
    patch_rows = cache["patch_rows"]
    patch_cols = cache["patch_cols"]
    probe_sub = cache["probe_sub"]
    cf0 = cache["cf0"]
    cf1 = cache["cf1"]

    for b in range(amp_out.shape[0]):
        amp = amp_out[b, 0]
        phase = phase_out[b, 0]
        amp_patches = amp[patch_rows, patch_cols]
        phase_patches = phase[patch_rows, patch_cols]
        psi = torch.complex(
            amp_patches * torch.cos(np.pi * phase_patches),
            amp_patches * torch.sin(np.pi * phase_patches),
        )
        exit_wave = psi * probe_sub.unsqueeze(0)
        detector_wave = torch.fft.fftshift(
            torch.fft.fftn(exit_wave * cf1.unsqueeze(0), dim=(1, 2)),
            dim=(1, 2),
        ) * cf0.unsqueeze(0)
        diffraction_patterns = detector_wave.abs().square()
        max_vals = diffraction_patterns.flatten(1).max(dim=1)[0].view(-1, 1, 1).clamp_min(1e-8)
        diffraction_patterns = diffraction_patterns / max_vals
        all_diffractions.append(diffraction_patterns)

    return torch.stack(all_diffractions, dim=0)


def run_unsupervised_optimization(model, input_tensor, input_center, params, forward_cache, num_epochs=30):
    losses = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)
    model.train()

    print("  执行无监督优化...")

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred_amp, pred_phase = model(input_tensor)
        pred_diffraction = forward_propagation_fast(pred_amp, pred_phase, forward_cache)
        loss_data = torch.norm(pred_diffraction - input_center, p=2)
        loss = loss_data
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        pred_amp, pred_phase = model(input_tensor)
    return {
        'obj_amp': pred_amp[0, 0, :, :].detach().cpu().numpy(),
        'obj_phase': pred_phase[0, 0, :, :].detach().cpu().numpy(),
        'losses': losses
    }


# --------------------------------------------------------------------------
# 5. 缺陷检测
# --------------------------------------------------------------------------
def align_intensity(source, target):
    s = source.flatten()
    t = target.flatten()
    ones = np.ones_like(s)
    A = np.column_stack([s, ones])
    coeffs, _, _, _ = np.linalg.lstsq(A, t, rcond=None)
    a, b = coeffs
    return a * source + b


def detect_defects(amp_optimized, amp_clean, base_threshold=0.07, min_area=4, adaptive_factor=2.0):
    aligned_clean = align_intensity(amp_clean, amp_optimized)
    diff = np.abs(amp_optimized - aligned_clean)
    diff_mean = float(np.mean(diff))
    adaptive_threshold = base_threshold + adaptive_factor * diff_mean
    binary_mask = diff > adaptive_threshold

    H, W = binary_mask.shape
    visited = np.zeros((H, W), dtype=bool)
    contours = []
    for y in range(H):
        for x in range(W):
            if binary_mask[y, x] and not visited[y, x]:
                contour = []
                stack = [(y, x)]
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    is_edge = False
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                if not binary_mask[ny, nx]:
                                    is_edge = True
                                    break
                        if is_edge:
                            break
                    if is_edge:
                        contour.append((cy, cx))
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                if binary_mask[ny, nx] and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    stack.append((ny, nx))
                if len(contour) >= 3:
                    contour_arr = np.array(contour)
                    cxm = np.mean(contour_arr[:, 1])
                    cym = np.mean(contour_arr[:, 0])
                    angles = np.arctan2(contour_arr[:, 0] - cym, contour_arr[:, 1] - cxm)
                    sort_idx = np.argsort(angles)
                    ordered = contour_arr[sort_idx]
                    if len(ordered) >= 3:
                        contours.append(ordered)

    defect_regions = []
    for contour in contours:
        area = len(contour)
        if area >= min_area:
            cx = int(np.mean(contour[:, 1]))
            cy = int(np.mean(contour[:, 0]))
            defect_regions.append({
                'center': (cy, cx),
                'area': area,
                'contour': contour,
            })
    return defect_regions, diff, aligned_clean, binary_mask


# --------------------------------------------------------------------------
# 6. 保存结果与可视化 (完整图片版本)
# --------------------------------------------------------------------------
def save_sample_plots(idx, results, label_data=None, region=None, pie_errors=None, uns_losses=None, noise_info=None):
    sample_dir = os.path.join(OUTPUT_DIR, f"sample_{idx:03d}")
    os.makedirs(sample_dir, exist_ok=True)

    if region is not None:
        x_min, x_max, y_min, y_max = region
    else:
        x_min, x_max, y_min, y_max = 0, results['original'][0].shape[0], 0, results['original'][0].shape[1]

    def crop(img):
        return img[y_min:y_max, x_min:x_max]

    for name, (amp, phase) in results.items():
        np.save(os.path.join(sample_dir, f"{name}_amp.npy"), amp.astype(np.float32))
        np.save(os.path.join(sample_dir, f"{name}_phase.npy"), phase.astype(np.float32))
        plt.imsave(os.path.join(sample_dir, f"{name}_amp.png"), amp, cmap='gray')
        plt.imsave(os.path.join(sample_dir, f"{name}_phase.png"), phase, cmap='viridis')

    if pie_errors is not None:
        for alg, errors in pie_errors.items():
            np.save(os.path.join(sample_dir, f"{alg}_errors.npy"), np.array(errors, dtype=np.float32))
            fig, axes = plt.subplots(figsize=(6, 3.5), dpi=150)
            axes.plot(range(1, len(errors) + 1), errors, marker='o', markersize=2.5, linewidth=1.5)
            axes.set_title(f"{alg.upper()} Fourier amp err — sample {idx} (with noise)")
            axes.set_xlabel("Iteration")
            axes.set_ylabel("Error")
            axes.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, f"{alg}_errors.png"), dpi=150)
            plt.close(fig)

    if uns_losses is not None:
        np.save(os.path.join(sample_dir, "unsupervised_loss.npy"), np.array(uns_losses, dtype=np.float32))
        fig, axes = plt.subplots(figsize=(6, 3.5), dpi=150)
        axes.plot(range(1, len(uns_losses) + 1), uns_losses, marker='s', markersize=2.5, linewidth=1.5, color='orange')
        axes.set_title(f"Unsupervised Loss — sample {idx} (with noise)")
        axes.set_xlabel("Epoch")
        axes.set_ylabel("Loss")
        axes.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, "unsupervised_loss.png"), dpi=150)
        plt.close(fig)

    mse_values = {}
    if label_data is not None:
        gt_defect = crop(label_data[0].astype(np.float32))
        gt_clean = crop(label_data[2].astype(np.float32))
        np.save(os.path.join(sample_dir, "label_defect_amp_roi.npy"), gt_defect)
        np.save(os.path.join(sample_dir, "label_clean_amp_roi.npy"), gt_clean)

        for name, (amp, phase) in results.items():
            amp_roi = crop(amp.astype(np.float32))
            mse_values[f"{name}_vs_defect_label"] = float(np.mean((amp_roi - gt_defect) ** 2))
            mse_values[f"{name}_vs_clean_label"] = float(np.mean((amp_roi - gt_clean) ** 2))

        with open(os.path.join(sample_dir, "mse_roi.txt"), 'w') as f:
            f.write("metric\tvalue\n")
            for k, v in mse_values.items():
                f.write(f"{k}\t{v:.8f}\n")
            if noise_info:
                f.write("\n")
                f.write("# 噪声参数:\n")
                if NOISE_CONFIG.get('add_poisson', False):
                    f.write(f"poisson_scale\t{NOISE_CONFIG['poisson_scale']}\n")
                if 'gaussian_snr_db' in noise_info:
                    f.write(f"gaussian_snr_db\t{noise_info['gaussian_snr_db']:.2f}\n")

    def show_roi_only(ax, img, title_text, vmin=None, vmax=None, cmap='gray'):
        roi_img = crop(img)
        if vmin is None:
            vmin = float(np.min(roi_img))
        if vmax is None:
            vmax = float(np.max(roi_img))
        ax.imshow(roi_img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title_text, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    def show_diff(ax, img_ref, label_clean_roi, title_text):
        diff = np.abs(crop(img_ref).astype(np.float32) - label_clean_roi)
        ax.imshow(diff, cmap='hot', vmin=0.0, vmax=float(np.percentile(diff, 99)) + 1e-6)
        ax.set_title(title_text, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    if label_data is not None:
        clean_label_roi = crop(label_data[2].astype(np.float32))

        full_panels = [
            ("Label (clean)", label_data[2].astype(np.float32)),
            ("Label (defect)", label_data[0].astype(np.float32)),
            ("Pretrained", results['original'][0].astype(np.float32)),
            ("Random Init", results['random_init'][0].astype(np.float32)),
            ("ePIE (with init)", results['epie_with_init'][0].astype(np.float32)),
            ("rPIE (with init)", results['rpie_with_init'][0].astype(np.float32)),
            ("mPIE (with init)", results['mpie_with_init'][0].astype(np.float32)),
            ("ePIE (random)", results['epie_random'][0].astype(np.float32)),
            ("rPIE (random)", results['rpie_random'][0].astype(np.float32)),
            ("mPIE (random)", results['mpie_random'][0].astype(np.float32)),
            ("Unsupervised", results['unsupervised'][0].astype(np.float32)),
        ]

        n_cols = 5
        n_rows = (len(full_panels) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), dpi=150)
        axes = axes.flatten()

        for i, (title, img) in enumerate(full_panels):
            show_roi_only(axes[i], img, title, cmap='gray')
        for i in range(len(full_panels), len(axes)):
            axes[i].axis('off')

        fig.suptitle(f"Sample {idx} — All Methods Amplitude Comparison (with noise)", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, "comparison_all_methods.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        pie_with_init_panels = [
            ("Label (clean)", label_data[2].astype(np.float32)),
            ("Label (defect)", label_data[0].astype(np.float32)),
            ("Pretrained", results['original'][0].astype(np.float32)),
            ("ePIE (with init)", results['epie_with_init'][0].astype(np.float32)),
            ("rPIE (with init)", results['rpie_with_init'][0].astype(np.float32)),
            ("mPIE (with init)", results['mpie_with_init'][0].astype(np.float32)),
            ("Unsupervised", results['unsupervised'][0].astype(np.float32)),
        ]

        n_cols_wi = 7
        fig, axes = plt.subplots(1, n_cols_wi, figsize=(4 * n_cols_wi, 4.5), dpi=150)
        for i, (title, img) in enumerate(pie_with_init_panels):
            show_roi_only(axes[i], img, title, cmap='gray')
        fig.suptitle(f"Sample {idx} — With Initial Prediction (with noise)", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, "comparison_with_init.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        pie_random_panels = [
            ("Label (clean)", label_data[2].astype(np.float32)),
            ("Label (defect)", label_data[0].astype(np.float32)),
            ("Random Init", results['random_init'][0].astype(np.float32)),
            ("ePIE (random)", results['epie_random'][0].astype(np.float32)),
            ("rPIE (random)", results['rpie_random'][0].astype(np.float32)),
            ("mPIE (random)", results['mpie_random'][0].astype(np.float32)),
        ]

        n_cols_r = 6
        fig, axes = plt.subplots(1, n_cols_r, figsize=(4 * n_cols_r, 4.5), dpi=150)
        for i, (title, img) in enumerate(pie_random_panels):
            show_roi_only(axes[i], img, title, cmap='gray')
        fig.suptitle(f"Sample {idx} — Random Initialization (with noise)", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, "comparison_random_init.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        diff_panels = [
            ("|Label(defect) − Label(clean)|", label_data[0].astype(np.float32)),
            ("|Pretrained − Label(clean)|", results['original'][0].astype(np.float32)),
            ("|ePIE(with init) − Label(clean)|", results['epie_with_init'][0].astype(np.float32)),
            ("|rPIE(with init) − Label(clean)|", results['rpie_with_init'][0].astype(np.float32)),
            ("|mPIE(with init) − Label(clean)|", results['mpie_with_init'][0].astype(np.float32)),
            ("|ePIE(random) − Label(clean)|", results['epie_random'][0].astype(np.float32)),
            ("|rPIE(random) − Label(clean)|", results['rpie_random'][0].astype(np.float32)),
            ("|mPIE(random) − Label(clean)|", results['mpie_random'][0].astype(np.float32)),
            ("|Unsupervised − Label(clean)|", results['unsupervised'][0].astype(np.float32)),
        ]

        n_cols_diff = 3
        n_rows_diff = 3
        fig, axes = plt.subplots(n_rows_diff, n_cols_diff, figsize=(5 * n_cols_diff, 5 * n_rows_diff), dpi=150)
        axes = axes.flatten()
        for i, (title, img) in enumerate(diff_panels):
            show_diff(axes[i], img, clean_label_roi, title)
        fig.suptitle(f"Sample {idx} — Difference Maps vs Clean Label (with noise)", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, "comparison_diff_maps.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        detection_methods = [
            ("Pretrained", results['original'][0].astype(np.float32)),
            ("ePIE (with init)", results['epie_with_init'][0].astype(np.float32)),
            ("rPIE (with init)", results['rpie_with_init'][0].astype(np.float32)),
            ("mPIE (with init)", results['mpie_with_init'][0].astype(np.float32)),
            ("ePIE (random)", results['epie_random'][0].astype(np.float32)),
            ("rPIE (random)", results['rpie_random'][0].astype(np.float32)),
            ("mPIE (random)", results['mpie_random'][0].astype(np.float32)),
            ("Unsupervised", results['unsupervised'][0].astype(np.float32)),
        ]

        clean_label = label_data[2].astype(np.float32)
        n_cols_det = 4
        n_rows_det = 2
        fig, axes = plt.subplots(n_rows_det, n_cols_det, figsize=(5 * n_cols_det, 4 * n_rows_det), dpi=150)
        axes = axes.flatten()

        for i, (title, amp_img) in enumerate(detection_methods):
            ax = axes[i]
            ax.imshow(crop(amp_img), cmap='gray')
            defects, diff_map, aligned_clean, binary = detect_defects(crop(amp_img), crop(clean_label))
            for d in defects:
                pts = d['contour']
                ax.plot(pts[:, 1], pts[:, 0], color='red', linewidth=0.8)
            ax.set_title(f"{title} — {len(defects)} defects", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"Defect Detection — sample {idx} (with noise)", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, "defect_detection_all.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        phase_panels = [
            ("Label (defect) phase", label_data[1].astype(np.float32)),
            ("Pretrained phase", results['original'][1].astype(np.float32)),
            ("ePIE (with init) phase", results['epie_with_init'][1].astype(np.float32)),
            ("rPIE (with init) phase", results['rpie_with_init'][1].astype(np.float32)),
            ("mPIE (with init) phase", results['mpie_with_init'][1].astype(np.float32)),
            ("ePIE (random) phase", results['epie_random'][1].astype(np.float32)),
            ("rPIE (random) phase", results['rpie_random'][1].astype(np.float32)),
            ("mPIE (random) phase", results['mpie_random'][1].astype(np.float32)),
            ("Unsupervised phase", results['unsupervised'][1].astype(np.float32)),
        ]

        n_cols_phase = 3
        n_rows_phase = 3
        fig, axes = plt.subplots(n_rows_phase, n_cols_phase, figsize=(5 * n_cols_phase, 5 * n_rows_phase), dpi=150)
        axes = axes.flatten()
        for i, (title, img) in enumerate(phase_panels):
            show_roi_only(axes[i], img, title, cmap='viridis')
        fig.suptitle(f"Phase Comparison — sample {idx} (with noise)", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, "comparison_phase_all.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    return mse_values


def save_summary(mse_rows, noise_config):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    headers = [
        'sample_index',
        'original_vs_defect_label',
        'epie_with_init_vs_defect_label',
        'rpie_with_init_vs_defect_label',
        'mpie_with_init_vs_defect_label',
        'epie_random_vs_defect_label',
        'rpie_random_vs_defect_label',
        'mpie_random_vs_defect_label',
        'unsupervised_vs_defect_label',
        'original_vs_clean_label',
        'epie_with_init_vs_clean_label',
        'rpie_with_init_vs_clean_label',
        'mpie_with_init_vs_clean_label',
        'epie_random_vs_clean_label',
        'rpie_random_vs_clean_label',
        'mpie_random_vs_clean_label',
        'unsupervised_vs_clean_label',
    ]

    txt_path = os.path.join(OUTPUT_DIR, "summary_mse.txt")
    with open(txt_path, 'w') as f:
        f.write(f"# 噪声参数配置:\n")
        f.write(f"# add_poisson={noise_config['add_poisson']}\n")
        f.write(f"# add_gaussian={noise_config['add_gaussian']}\n")
        if noise_config['add_poisson']:
            f.write(f"# poisson_scale={noise_config['poisson_scale']}\n")
        if noise_config['add_gaussian']:
            f.write(f"# gaussian_snr_db={noise_config['gaussian_snr_db']}\n")
        f.write(f"# seed={noise_config['seed']}\n")
        f.write(f"#\n")
        f.write('\t'.join(headers) + '\n')
        for row in mse_rows:
            vals = [str(row.get(h, float('nan'))) for h in headers]
            f.write('\t'.join(vals) + '\n')

        f.write('\n')
        f.write('# 汇总统计 (mean / std / median / min / max / n)  across all samples:\n')
        for k in headers[1:]:
            vals = [row[k] for row in mse_rows if k in row and not (isinstance(row[k], float) and row[k] != row[k])]
            if not vals:
                continue
            vals = np.array(vals, dtype=np.float64)
            f.write(f"{k}\tmean={np.mean(vals):.8f}\tstd={np.std(vals):.8f}\tmedian={np.median(vals):.8f}\tmin={np.min(vals):.8f}\tmax={np.max(vals):.8f}\tn={len(vals)}\n")

    np.save(os.path.join(OUTPUT_DIR, "summary_mse.npy"), np.array(mse_rows, dtype=object))
    print(f"  [saved] {txt_path}")

    fig, axes = plt.subplots(figsize=(12, 6), dpi=150)
    data_box = [
        [row['original_vs_defect_label'] for row in mse_rows],
        [row['epie_with_init_vs_defect_label'] for row in mse_rows],
        [row['rpie_with_init_vs_defect_label'] for row in mse_rows],
        [row['mpie_with_init_vs_defect_label'] for row in mse_rows],
        [row['epie_random_vs_defect_label'] for row in mse_rows],
        [row['rpie_random_vs_defect_label'] for row in mse_rows],
        [row['mpie_random_vs_defect_label'] for row in mse_rows],
        [row['unsupervised_vs_defect_label'] for row in mse_rows],
    ]
    labels = ['Pretrained', 'ePIE\n(with init)', 'rPIE\n(with init)', 'mPIE\n(with init)',
              'ePIE\n(random)', 'rPIE\n(random)', 'mPIE\n(random)', 'Unsupervised']
    bp = axes.boxplot(data_box, labels=labels, patch_artist=True,
                       medianprops={'color': 'orange', 'linewidth': 1.5}, widths=0.5)
    colors = ['lightblue', 'lightgreen', 'lightgreen', 'lightgreen',
              'salmon', 'salmon', 'salmon', 'orange']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1.0)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1.0)
    axes.set_title(f"MSE vs defect label ({len(mse_rows)} samples, with noise)", fontsize=13)
    axes.set_ylabel("MSE", fontsize=11)
    axes.grid(True, axis='y', alpha=0.3, linestyle='--')
    axes.tick_params(axis='both', labelsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mse_boxplot_vs_defect.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(12, 6), dpi=150)
    data_box2 = [
        [row['epie_with_init_vs_clean_label'] for row in mse_rows],
        [row['rpie_with_init_vs_clean_label'] for row in mse_rows],
        [row['mpie_with_init_vs_clean_label'] for row in mse_rows],
        [row['epie_random_vs_clean_label'] for row in mse_rows],
        [row['rpie_random_vs_clean_label'] for row in mse_rows],
        [row['mpie_random_vs_clean_label'] for row in mse_rows],
        [row['unsupervised_vs_clean_label'] for row in mse_rows],
    ]
    labels2 = ['ePIE\n(with init)', 'rPIE\n(with init)', 'mPIE\n(with init)',
               'ePIE\n(random)', 'rPIE\n(random)', 'mPIE\n(random)', 'Unsupervised']
    bp = axes.boxplot(data_box2, labels=labels2, patch_artist=True,
                       medianprops={'color': 'orange', 'linewidth': 1.5}, widths=0.5)
    colors2 = ['lightgreen', 'lightgreen', 'lightgreen', 'salmon', 'salmon', 'salmon', 'orange']
    for patch, color in zip(bp['boxes'], colors2):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1.0)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1.0)
    axes.set_title(f"MSE vs clean label ({len(mse_rows)} samples, with noise)", fontsize=13)
    axes.set_ylabel("MSE", fontsize=11)
    axes.grid(True, axis='y', alpha=0.3, linestyle='--')
    axes.tick_params(axis='both', labelsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mse_boxplot_vs_clean.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(16, 7), dpi=150)
    xs = list(range(len(mse_rows)))
    series_map = [
        ('Pretrained', 'original_vs_defect_label', 'steelblue', 'o'),
        ('ePIE (with init)', 'epie_with_init_vs_defect_label', 'green', 's'),
        ('rPIE (with init)', 'rpie_with_init_vs_defect_label', 'limegreen', '^'),
        ('mPIE (with init)', 'mpie_with_init_vs_defect_label', 'seagreen', 'D'),
        ('ePIE (random)', 'epie_random_vs_defect_label', 'red', 's'),
        ('rPIE (random)', 'rpie_random_vs_defect_label', 'orangered', '^'),
        ('mPIE (random)', 'mpie_random_vs_defect_label', 'coral', 'D'),
        ('Unsupervised', 'unsupervised_vs_defect_label', 'orange', 'X'),
    ]
    for name, key, color, marker in series_map:
        axes.plot(xs, [row[key] for row in mse_rows], marker=marker, markersize=3.5,
                   linewidth=1.3, label=name, color=color, alpha=0.85)
    axes.set_title(f"Per-sample MSE (vs defect label) — {len(mse_rows)} samples (with noise)", fontsize=13)
    axes.set_xlabel("Sample index", fontsize=11)
    axes.set_ylabel("MSE", fontsize=11)
    axes.legend(fontsize=9, loc='upper right')
    axes.grid(True, alpha=0.3, linestyle='--')
    axes.tick_params(axis='both', labelsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mse_per_sample_vs_defect.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  [saved] {os.path.join(OUTPUT_DIR, 'mse_boxplot_vs_defect.png')}")
    print(f"  [saved] {os.path.join(OUTPUT_DIR, 'mse_per_sample_vs_defect.png')}")


# --------------------------------------------------------------------------
# 7. 主循环 (30 个样本)
# --------------------------------------------------------------------------
def main():
    global params
    t0 = time.time()

    params = PtychoParams()
    params.overlap = 0.7
    params.initialize()
    print("\n参数配置 (overlap=0.7):")
    print(f"  Nd: {params.Nd}")
    print(f"  No: {params.No}")
    print(f"  step: {params.step}")
    print(f"  probesize: {params.probesize}")
    print(f"  overlap: {params.overlap}")

    illumination_region = compute_illumination_region(params)
    print(f"  照明区域: x[{illumination_region[0]}:{illumination_region[1]}], "
          f"y[{illumination_region[2]}:{illumination_region[3]}]")

    input_files = sorted(glob.glob(os.path.join(DATA_DIR, "input_*.npy")))
    if not input_files:
        print(f"错误：未找到测试数据 {DATA_DIR}")
        return

    num_samples_total = len(input_files)
    num_samples = min(SAMPLE_CONFIG['num_samples'], num_samples_total)
    
    if SAMPLE_CONFIG['random_select']:
        rng = np.random.default_rng(seed=SAMPLE_CONFIG['seed'])
        sample_indices = sorted(rng.choice(num_samples_total, size=num_samples, replace=False).tolist())
        print(f"\n找到 {num_samples_total} 个测试样本，随机抽取 {num_samples} 个")
    else:
        sample_indices = list(range(num_samples))
        print(f"\n找到 {num_samples_total} 个测试样本，按顺序抽取前 {num_samples} 个")
    print(f"样本索引: {sample_indices}")

    model_template = TransUNetPtychoPadded(img_size=128, in_channels=100, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4.0)
    model_template = load_pretrained_weights(model_template, MODEL_PATH)
    model_template.to(DEVICE)

    mse_rows = []
    input_size = 128
    original_size = 64
    pad_start = (input_size - original_size) // 2
    pad_end = pad_start + original_size

    positions_rc = prepare_positions_for_pie(params, device=DEVICE)
    cf0, cf1 = build_propagators(params, device=DEVICE)
    uns_forward_cache = build_unsupervised_forward_cache(params)
    Nd_local = params.Nd[0]
    probe_small = params.probe[
        (params.No - Nd_local) // 2:(params.No - Nd_local) // 2 + Nd_local,
        (params.No - Nd_local) // 2:(params.No - Nd_local) // 2 + Nd_local,
    ].contiguous().to(DEVICE)

    pie_configs = {
        "epie": PIEConfig(algorithm="epie", iterations=50, beta_object=0.8),
        "rpie": PIEConfig(algorithm="rpie", iterations=50, beta_object=0.8, alpha_object=0.1),
        "mpie": PIEConfig(algorithm="mpie", iterations=50, beta_object=0.8, alpha_object=0.1,
                          momentum_friction=0.7, momentum_feedback=0.3, momentum_start=5),
    }

    for loop_i, sample_idx in enumerate(sample_indices):
        print(f"\n{'=' * 60}")
        print(f"[{loop_i + 1}/{num_samples}] dataset_idx={sample_idx} [{os.path.basename(input_files[sample_idx])}]")
        print('=' * 60)

        input_data = np.load(input_files[sample_idx])
        print(f"  原始输入数据形状: {input_data.shape}")

        input_data_noisy, noise_info = add_noise_to_diffraction(input_data, NOISE_CONFIG)
        print(f"  已添加噪声:")
        if NOISE_CONFIG.get('add_poisson', False):
            print(f"    泊松噪声 scale: {NOISE_CONFIG['poisson_scale']}")
        else:
            print(f"    泊松噪声: 未添加")
        if 'gaussian_snr_db' in noise_info:
            print(f"    高斯噪声 SNR: {noise_info['gaussian_snr_db']:.1f} dB")
        else:
            print(f"    高斯噪声: 未添加")

        input_tensor = torch.from_numpy(input_data_noisy).float().unsqueeze(0).to(DEVICE)
        input_center = input_tensor[:, :, pad_start:pad_end, pad_start:pad_end]

        label_data = None
        label_file_path = os.path.join(LABEL_DIR, os.path.basename(input_files[sample_idx]).replace('input', 'label'))
        if os.path.exists(label_file_path):
            label_data = np.load(label_file_path)
            print(f"  标签：{os.path.basename(label_file_path)}, 形状：{label_data.shape}")

        model_template.eval()
        with torch.no_grad():
            original_amp_t, original_phase_t = model_template(input_tensor)
        original_amp = original_amp_t[0, 0, :, :].detach().cpu().numpy()
        original_phase = original_phase_t[0, 0, :, :].detach().cpu().numpy()

        obj_init = torch.complex(
            torch.from_numpy(original_amp * np.cos(np.pi * original_phase)).float(),
            torch.from_numpy(original_amp * np.sin(np.pi * original_phase)).float(),
        ).to(DEVICE)

        random_amp, random_phase, obj_init_random = create_random_initial_guess(params.No, seed=sample_idx)
        print(f"  随机初始振幅范围: [{random_amp.min():.4f}, {random_amp.max():.4f}]")

        diff_cropped = input_data_noisy[:, pad_start:pad_end, pad_start:pad_end]
        measured_amp = torch.from_numpy(np.sqrt(np.clip(diff_cropped, 0.0, None))).to(DEVICE)

        pie_results = {}
        pie_errors = {}

        print("  --- 有初始预测的 PIE 算法 ---")
        for alg, cfg in pie_configs.items():
            print(f"  [{alg.upper()}] 开始 {cfg.iterations} 次迭代（有初始预测）...")
            t_s = time.time()
            amp_np, phase_np, errors = run_pie_iteration(
                obj_init, probe_small, positions_rc, measured_amp, cf0, cf1, cfg,
            )
            print(f"  [{alg.upper()}] 完成，耗时 {time.time() - t_s:.2f}s，最后误差 {errors[-1]:.6f}")
            pie_results[f"{alg}_with_init"] = (amp_np, phase_np)
            pie_errors[f"{alg}_with_init"] = errors

        print("  --- 无初始预测的 PIE 算法 ---")
        for alg, cfg in pie_configs.items():
            print(f"  [{alg.upper()}] 开始 {cfg.iterations} 次迭代（无初始预测）...")
            t_s = time.time()
            amp_np, phase_np, errors = run_pie_iteration(
                obj_init_random, probe_small, positions_rc, measured_amp, cf0, cf1, cfg,
            )
            print(f"  [{alg.upper()}] 完成，耗时 {time.time() - t_s:.2f}s，最后误差 {errors[-1]:.6f}")
            pie_results[f"{alg}_random"] = (amp_np, phase_np)
            pie_errors[f"{alg}_random"] = errors

        model_u = TransUNetPtychoPadded(img_size=128, in_channels=100, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4.0)
        model_u = load_pretrained_weights(model_u, MODEL_PATH)
        model_u.to(DEVICE)
        t_s = time.time()
        uns_result = run_unsupervised_optimization(
            model_u,
            input_tensor,
            input_center,
            params,
            uns_forward_cache,
            num_epochs=30,
        )
        print(f"  unsupervised (30 epochs) 完成，耗时 {time.time() - t_s:.2f}s，最后 loss: {uns_result['losses'][-1]:.6f}")

        results = {
            'original': (original_amp, original_phase),
            'random_init': (random_amp, random_phase),
            **pie_results,
            'unsupervised': (uns_result['obj_amp'], uns_result['obj_phase']),
        }

        mse_values = save_sample_plots(
            loop_i, results,
            label_data=label_data,
            region=illumination_region,
            pie_errors=pie_errors,
            uns_losses=uns_result['losses'],
            noise_info=NOISE_CONFIG,
        )

        if label_data is not None:
            row = {'sample_index': loop_i}
            row.update(mse_values)
            mse_rows.append(row)
            print(f"  ROI MSE vs defect:")
            for name in ['original', 'epie_with_init', 'rpie_with_init', 'mpie_with_init',
                         'epie_random', 'rpie_random', 'mpie_random', 'unsupervised']:
                val = row.get(f'{name}_vs_defect_label', float('nan'))
                print(f"    {name}: {val:.6f}")
            print(f"  ROI MSE vs clean:")
            for name in ['original', 'epie_with_init', 'rpie_with_init', 'mpie_with_init',
                         'epie_random', 'rpie_random', 'mpie_random', 'unsupervised']:
                val = row.get(f'{name}_vs_clean_label', float('nan'))
                print(f"    {name}: {val:.6f}")

    save_summary(mse_rows, NOISE_CONFIG)
    elapsed = time.time() - t0
    print(f"\n全部完成，总耗时 {elapsed:.1f}s。结果保存在 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()