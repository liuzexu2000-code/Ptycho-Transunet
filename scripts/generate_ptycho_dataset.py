import numpy as np
import os
import torch
from PIL import Image
from tqdm import tqdm
import shutil
import random

class Parameters:
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
        self.overlap = 0.25
        self.step = None
        self.No = None
        self.Np = None
        self.probe = None

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

params = Parameters()
params.initialize()

def create_probe(params):
    probe_size_pixels = int(params.probesize / params.dxp)
    probe = torch.zeros(params.Np)
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
    
    return probe

params.probe = create_probe(params)

probe_np = params.probe.numpy()
probe_image = (probe_np * 255).astype(np.uint8)
probe_img = Image.fromarray(probe_image)
probe_img.save('d:/BaiduSyncdisk/second_term/Uptynet/article_experiment/probe.png')
print(f"探针图片已保存: probe.png")

def Frau_create(elem):
    Psi, probe, complex_fourier0, complex_fourier1 = elem
    Psi_ = probe * Psi
    F_ = torch.fft.fftn(Psi_ * complex_fourier1, dim=(0,1))
    F = complex_fourier0 * torch.fft.fftshift(F_, dim=(0,1))
    return F

def generate_diffraction_pattern(amp, phase, probe, params):
    real = amp * torch.cos(2 * np.pi * phase)
    imag = amp * torch.sin(2 * np.pi * phase)
    Psi = torch.complex(real, imag)

    lambda_ = params.wavelength
    z = torch.tensor(params.z0, dtype=torch.cfloat)
    dx = params.dxd * params.bin
    L = dx * params.Nd[0]
    pi = torch.tensor(3.1415926535, dtype=torch.cfloat)
    k = 2 * pi / lambda_

    x = torch.arange(-L / 2, L / 2, dx)
    dx0 = params.dxo
    L0 = dx0 * params.Nd[0]
    x0 = torch.arange(-L0 / 2, L0 / 2, dx0)
    X, Y = torch.meshgrid([x, x], indexing='xy')
    X0, Y0 = torch.meshgrid([x0, x0], indexing='xy')

    delta0 = k / (2 * z) * (X ** 2 + Y ** 2)
    delta1 = k / (2 * z) * (X0 ** 2 + Y0 ** 2)
    complex_fourier0 = dx0 * dx0 * 1 / (1j * lambda_ * z) * (torch.cos(delta0) + 1j * torch.sin(delta0))
    complex_fourier1 = (torch.cos(delta1) + 1j * torch.sin(delta1))

    elem = (Psi, probe, complex_fourier0, complex_fourier1)
    Psi2 = Frau_create(elem)

    x = torch.abs(Psi2) ** 2
    max_val_new = x.max()
    if max_val_new > 0:
        x = x / max_val_new

    return x.float()

def generate_ptycho_dataset(image_path, output_dir, num_samples=3000):
    probe = params.probe
    grid_size_x = params.number_diff_x
    grid_size_y = params.number_diff_y
    num_diffraction = grid_size_x * grid_size_y
    patch_size = params.Nd[0]
    step_size = params.step
    full_size = params.No

    print(f"完整物体尺寸: {full_size}×{full_size}")
    print(f"衍射图数量: {num_diffraction} ({grid_size_x}×{grid_size_y})")
    print(f"每张衍射图尺寸: {patch_size}×{patch_size}")
    print(f"扫描步距: {step_size}")
    print(f"探针大小: {params.probesize*1e6:.2f} μm ({int(params.probesize/params.dxp)} pixels)")
    print(f"重叠率: {params.overlap*100:.1f}%")

    os.makedirs(os.path.join(output_dir, 'train', 'inputs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'inputs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'labels'), exist_ok=True)

    train_count = 0
    test_count = 0

    if os.path.exists(image_path):
        img = Image.open(image_path)
        img_np = np.array(img, dtype=np.float32) / 255.0
        
        if img_np.size == 0:
            print("错误: 图像数组为空!")
            return
        
        if len(img_np.shape) == 3:
            img_np = np.mean(img_np, axis=2)
            print("转换为灰度图像")
        
        height, width = img_np.shape
        print(f"加载图像: {os.path.basename(image_path)}, 原始尺寸: {height}×{width}")
    else:
        print(f"错误: 图像不存在: {image_path}")
        return

    print(f"图像高度: {height}, 宽度: {width}")
    print(f"训练集比例: 75%, 测试集比例: 25%")
    print(f"训练区域: 左上3/4, 测试区域: 右下1/4")

    train_samples = int(num_samples * 0.75)
    test_samples = num_samples - train_samples

    half_height = height // 2
    half_width = width // 2

    used_positions = set()

    for sample_idx in tqdm(range(num_samples), desc="生成叠层成像序列数据"):
        is_train = sample_idx < train_samples

        while True:
            if is_train:
                quadrant = random.randint(0, 2)
                if quadrant == 0:
                    y_offset = random.randint(0, max(half_height - full_size, 0))
                    x_offset = random.randint(0, max(half_width - full_size, 0))
                elif quadrant == 1:
                    y_offset = random.randint(0, max(half_height - full_size, 0))
                    x_offset = random.randint(half_width, max(width - full_size, half_width))
                else:
                    y_offset = random.randint(half_height, max(height - full_size, half_height))
                    x_offset = random.randint(0, max(half_width - full_size, 0))
            else:
                y_offset = random.randint(half_height, max(height - full_size, half_height))
                x_offset = random.randint(half_width, max(width - full_size, half_width))

            pos_key = (y_offset, x_offset)
            if pos_key not in used_positions:
                used_positions.add(pos_key)
                break

        full_obj = img_np[y_offset:y_offset+full_size, x_offset:x_offset+full_size]
        
        rotation_angle = random.choice([0, 90, -90, 180])
        if rotation_angle != 0:
            full_obj = np.rot90(full_obj, k=rotation_angle // 90).copy()
        
        if random.random() < 0.5:
            full_obj = np.fliplr(full_obj).copy()
        
        if random.random() < 0.5:
            full_obj = np.flipud(full_obj).copy()

        diffraction_patterns = []
        for row in range(grid_size_y):
            for col in range(grid_size_x):
                y_start = (grid_size_y - row - 1) * step_size
                x_start = col * step_size

                y_end = y_start + patch_size
                x_end = x_start + patch_size

                amp_patch = full_obj[y_start:y_end, x_start:x_end]
                phase_patch = amp_patch

                amp_tensor = torch.from_numpy(amp_patch).float()
                phase_tensor = torch.from_numpy(phase_patch).float()

                diffraction = generate_diffraction_pattern(amp_tensor, phase_tensor, probe, params)
                diffraction_patterns.append(diffraction.numpy())

        diffraction_stack = np.stack(diffraction_patterns)

        full_amp = full_obj
        full_phase = full_obj

        label = np.stack([full_amp, full_phase])

        if is_train:
            data_dir = 'train'
            idx = train_count
            train_count += 1
        else:
            data_dir = 'test'
            idx = test_count
            test_count += 1

        input_path = os.path.join(output_dir, data_dir, 'inputs', f'input_{idx:06d}.npy')
        label_path = os.path.join(output_dir, data_dir, 'labels', f'label_{idx:06d}.npy')
        np.save(input_path, diffraction_stack)
        np.save(label_path, label)

    print(f"\n数据集生成完成!")
    print(f"训练样本: {train_count}")
    print(f"测试样本: {test_count}")
    print(f"输入尺寸: {num_diffraction}×{patch_size}×{patch_size}")
    print(f"标签尺寸: 2×{full_size}×{full_size}")

if __name__ == '__main__':
    image_path = 'd:/BaiduSyncdisk/second_term/Uptynet/original_images/object/grid.png'
    output_dir = 'd:/BaiduSyncdisk/second_term/Uptynet/article_experiment/data_ptycho_overlap025'
    num_samples = 5000
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    generate_ptycho_dataset(image_path, output_dir, num_samples=num_samples)