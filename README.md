# Ptycho-TransUNet: 叠层成像与Transformer UNet相结合的相位恢复

## 简介

这是一个将叠层成像（Ptychography）与TransUNet相结合的深度学习模型，用于从衍射图案中重构物体的振幅和相位。本项目包含数据集生成、模型训练、推理可视化等完整流程。

## 特点

- **TransUNet架构**: 结合了UNet的编码器-解码器结构和Transformer的全局注意力机制
- **多重叠率数据集**: 支持 overlap=0, 0.25, 0.5, 0.7 等多种配置
- **完整流程**: 从数据生成、模型训练到可视化分析
- **多种方法对比**: 包括 PtychoNN、TransUNet (单帧) 和 Ptycho-TransUNet (叠层)
- **芯片缺陷检测**: 支持可见光和EUV波段的芯片叠层成像缺陷检测
- **联合细化算法**: 结合传统PIE算法与无监督深度学习细化，实现带噪声条件下的高精度缺陷检测

## 文件夹结构

```
ptycho-transunet-github/
├── README.md                    # 说明文档
├── requirements.txt             # Python依赖包
├── scripts/                     # Python脚本
│   ├── generate_ptycho_dataset.py        # 生成叠层成像数据集
│   ├── train_ptycho_model.py             # 训练叠层成像模型
│   ├── train_single_models.py            # 训练单帧重构模型 (PtychoNN, TransUNet)
│   ├── compute_center_pattern_mse_distribution_overlap0.py    # 计算MSE分布
│   ├── compute_center_pattern_mse_distribution_overlap025.py
│   ├── compute_center_pattern_mse_distribution_overlap05.py
│   ├── compute_center_pattern_mse_distribution_overlap07.py
│   ├── compare_overlap_results.py        # 对比不同重叠率的结果
│   ├── plot_network_structure.py         # 绘制网络结构图
│   ├── visualize_ptycho_data.py          # 可视化数据
│   ├── visualize_single_data.py          # 可视化单帧数据
│   ├── test_model_load.py                # 模型加载测试
│   ├── generate_reconstruction_comparison_decreasing_mse.py   # 生成对比图
│   ├── generate_chip_patterns.py         # 生成芯片检测样本（大尺寸+缺陷）
│   ├── generate_ptycho_dataset_chip_padded.py        # 芯片叠层数据集生成（可见光，带padding）
│   ├── generate_ptycho_dataset_chip_padded_EUV.py    # 芯片叠层数据集生成（EUV，带padding）
│   ├── train_ptycho_chip_model_padded.py             # 训练芯片叠层模型（可见光）
│   ├── train_EUVptycho_chip_model_padded.py          # 训练芯片叠层模型（EUV）
│   └── combined_refinement_defect_30samples_noisy_fast_uns.py  # 联合细化与缺陷检测（带噪声）
├── checkpoints/                 # 预训练模型权重
│   ├── models/                  # Overlap=0.7 的模型
│   │   ├── best_transunet-ptycho_ptycho.pth    # Ptycho-TransUNet最佳模型
│   │   ├── best_ptychonn_single.pth             # PtychoNN最佳模型
│   │   ├── best_transunet_single.pth            # TransUNet最佳模型
│   │   └── ...
│   ├── models_overlap0/         # Overlap=0 的模型
│   ├── models_overlap025/       # Overlap=0.25 的模型
│   └── models_overlap05/        # Overlap=0.5 的模型
├── original_patterns/           # 原始物体图案 (用于生成数据集)
│   └── ... (各种网格图案PNG文件)
├── data_generation/             # 数据生成配置 (可选)
└── docs/                        # 文档 (可选)
```

## 安装

### 依赖环境

```bash
# 克隆仓库
git clone <repo-url>
cd ptycho-transunet-github

# 创建虚拟环境 (推荐)
conda create -n ptycho python=3.9
conda activate ptycho

# 安装依赖
pip install -r requirements.txt
```

### 主要依赖包

- torch >= 1.8
- torchvision
- numpy
- matplotlib
- pillow
- tqdm
- pandas

## 使用方法

### 0. 生成芯片检测样本（可选）

如果需要用于芯片检测的数据集，首先生成大尺寸(2048×2048)的芯片图案：

```bash
cd scripts
python generate_chip_patterns.py
```

这将生成包含正常和有缺陷的芯片图案，详见 [README_CHIP_GENERATION.md](./README_CHIP_GENERATION.md)

### 1. 生成叠层成像数据集

```bash
cd scripts

# 编辑 generate_ptycho_dataset.py 中的参数
# - image_path: 原始图案路径（可以使用生成的芯片图案）
# - output_dir: 输出文件夹
# - num_samples: 样本数量
# - overlap: 重叠率 (0, 0.25, 0.5, 0.7)

python generate_ptycho_dataset.py
```

### 2. 训练单帧模型 (PtychoNN, TransUNet)

```bash
python train_single_models.py
```

### 3. 训练叠层成像模型 (Ptycho-TransUNet)

```bash
# 为不同重叠率分别训练
# 编辑 train_ptycho_model.py 中的配置参数
python train_ptycho_model.py
```

### 4. 推理与可视化

```bash
# 可视化数据
python visualize_ptycho_data.py
python visualize_single_data.py

# 生成重构对比图
python generate_reconstruction_comparison_decreasing_mse.py

# 计算MSE分布并可视化
python compute_center_pattern_mse_distribution_overlap07.py
# ... 其他重叠率
python compare_overlap_results.py
```

### 5. 使用预训练模型

```python
import torch
from train_ptycho_model import create_ptycho_model

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_ptycho_model(output_size=128).to(device)
checkpoint = torch.load('checkpoints/models/best_transunet-ptycho_ptycho.pth', 
                         map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# 推理
# input_data: [batch, 100, 64, 64] - 100张衍射图案序列
with torch.no_grad():
    pred_amp, pred_phase = model(input_data)
```

### 6. 芯片缺陷检测应用

本项目支持在可见光和EUV（极紫外）波段下进行芯片缺陷检测，包含数据集生成、模型训练和联合细化缺陷检测的完整流程。

#### 6.1 芯片叠层成像数据集生成

**可见光波段（632.8 nm）:**

```bash
cd scripts

# 编辑 generate_ptycho_dataset_chip_padded.py 中的参数：
# - image_path: 带缺陷芯片图案路径
# - original_image_path: 无缺陷芯片图案路径（用于生成标签）
# - output_dir: 输出目录
# - num_samples: 样本数量
# - overlap: 重叠率（默认0.7）

python generate_ptycho_dataset_chip_padded.py
```

该脚本生成带padding的叠层成像数据集，标签包含4个通道：带缺陷振幅、带缺陷相位、无缺陷振幅、无缺陷相位。

**EUV波段（13.5 nm）:**

```bash
python generate_ptycho_dataset_chip_padded_EUV.py
```

EUV版本使用与可见光相同的数据结构，但物理参数针对EUV光刻场景进行了调整：
- 波长: 13.5 nm
- 探测器像素尺寸: 176 μm
- 探针大小: 0.162 μm
- 传播距离: 5.7 mm

#### 6.2 训练芯片缺陷检测模型

**可见光波段:**

```bash
python train_ptycho_chip_model_padded.py
```

**EUV波段:**

```bash
python train_EUVptycho_chip_model_padded.py
```

训练脚本的特点：
- 使用 **前向传播损失（Forward Loss）** 作为物理约束，将预测的振幅/相位通过正向衍射传播与原始衍射图对比
- 损失函数 = 监督损失（MSE） + λ × 前向传播损失
- 支持 padding 到能被8整除的尺寸，适配 TransUNet 的下采样需求
- 自动保存最佳模型权重和训练损失曲线

#### 6.3 联合细化与缺陷检测

```bash
python combined_refinement_defect_30samples_noisy_fast_uns.py
```

该脚本实现了多种重构算法的对比与缺陷检测，流程如下：

1. **加载预训练 TransUNet** 得到初始重构
2. **添加噪声**（泊松噪声 + 高斯噪声）模拟真实测量条件
3. **传统 PIE 算法细化**（ePIE、rPIE、mPIE），分别基于初始预测和随机初始化
4. **无监督深度学习细化**（向量化前向传播，快速版本）
5. **缺陷检测**：将重构结果与无缺陷参考图像对比，自动检测并标记缺陷区域

输出包括：
- 各方法重构结果对比图（振幅、相位）
- 缺陷检测结果图（红色轮廓标记缺陷位置）
- MSE统计箱线图和逐样本MSE曲线
- PIE迭代误差曲线和无监督损失曲线

## 网络架构

### TransUNet-Ptycho

```
Input: 100×64×64 (100张衍射图案)
  ↓
Encoder (ResNet blocks + MaxPool)
  ↓
TransUNet (Position embedding + Transformer blocks)
  ↓
Decoder (Upsample + Skip connections)
  ↓
Output: 2×H×W (振幅 + 相位)
```

### PtychoNN

```
Input: 1×64×64 (单张衍射图案)
  ↓
Encoder (3层 Conv2d + MaxPool)
  ↓
Decoder (3层 Upsample + Conv2d)
  ↓
Output: 2×64×64 (振幅 + 相位)
```

## 预训练模型下载

本仓库已包含预训练模型权重，位于 `checkpoints/` 文件夹中：

- **Overlap=0.7 (models/)**: 最高重叠率，最佳重构质量
- **Overlap=0.5 (models_overlap05/)**
- **Overlap=0.25 (models_overlap025/)**
- **Overlap=0 (models_overlap0/)**: 无重叠，单帧与叠层对比

## 结果展示

### 不同重叠率的重构效果

| Overlap | 照明区域 MSE (振幅) | 完整图像 MSE (振幅) | 改进率 |
|---------|---------------------|---------------------|--------|
| 0.7     | 0.012               | 0.034               | 64.7%  |
| 0.5     | 0.018               | 0.041               | 56.1%  |
| 0.25    | 0.025               | 0.052               | 51.9%  |
| 0       | 0.058               | 0.083               | 30.1%  |

### 性能指标

| 方法 | 平均 MSE | 推理时间 |
|------|----------|----------|
| PtychoNN | 0.087 | 0.2 ms |
| TransUNet | 0.061 | 0.4 ms |
| Ptycho-TransUNet (Ours) | **0.023** | 15 ms |

## 引用

如果您使用了本项目的代码或模型，请引用：

```
@article{
}
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请联系：
- Email: <zexu_liu@siom.ac.cn>
