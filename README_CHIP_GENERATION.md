
# 芯片检测样本生成说明

本项目提供了用于芯片检测的大尺寸样本生成方法，基于论文"Learning Photomask Distribution using Likelihood-based Generative Models"的思想。

## 功能特点

- **大尺寸支持**：支持生成2048×2048及更大尺寸的芯片图案
- **多种缺陷类型**：包含点缺陷、线缺陷、区域缺陷、缺失缺陷、多余缺陷、桥接缺陷等多种常见芯片缺陷
- **多种基础图案**：支持网格、线条、接触孔、混合图案等多种芯片风格
- **灵活配置**：可自定义样本数量、尺寸、缺陷类型和数量

## 使用方法

### 基本使用

```python
from scripts.generate_chip_patterns import ChipPatternGenerator, generate_chip_dataset

# 生成数据集
generate_chip_dataset(
    output_dir='./chip_patterns',
    num_samples=50,
    image_size=2048,
    add_defects=True
)
```

### 单独使用生成器

```python
generator = ChipPatternGenerator(image_size=2048)

# 生成基础图案
base_pattern = generator.generate_base_pattern(pattern_type='mixed')

# 添加缺陷
defect_pattern = base_pattern.copy()
defect_info = generator.add_defects(defect_pattern, defect_type='all', num_defects=20)
```

## 支持的图案类型

| 类型 | 说明 |
|------|------|
| `grid` | 网格图案（类似接触孔或金属层布局） |
| `line` | 线条图案（类似金属布线） |
| `contact` | 接触孔图案 |
| `mixed` | 混合多种元素的图案 |

## 支持的缺陷类型

| 缺陷类型 | 说明 |
|----------|------|
| `point` | 点缺陷（局部异常） |
| `line` | 线缺陷（断裂或多余线条） |
| `area` | 区域缺陷（大面积异常） |
| `missing` | 缺失缺陷（图案丢失） |
| `extra` | 多余缺陷（不该有的图案） |
| `bridge` | 桥接缺陷（不该连接的地方连接了） |

## 输出结构

```
chip_patterns/
├── normal/              # 正常样本
│   ├── normal_0000.png
│   ├── normal_0001.png
│   └── ...
├── defect/              # 缺陷样本
│   ├── defect_0000.png
│   ├── defect_0000_info.txt  # 缺陷信息
│   └── ...
└── visualizations/      # 可视化结果
    └── chip_patterns_comparison.png
```

## 示例

运行主程序可以生成测试样本：

```bash
python scripts/generate_chip_patterns.py
```

这将生成50个2048×2048的样本，并在visualizations目录下生成对比图。

## 与现有数据集的结合

生成的芯片图案可以直接用于现有的叠层成像数据集生成流程：

```python
from scripts.generate_ptycho_dataset import generate_ptycho_dataset

# 使用生成的芯片图案作为原始图像
generate_ptycho_dataset(
    image_path='./chip_patterns/normal/normal_0000.png',
    output_dir='./chip_ptycho_data',
    num_samples=1000
)
```

## 参考论文

本方法参考了论文中的思想：
- 学习光刻掩模的分布特征
- 生成真实的芯片图案
- 加入真实物理缺陷

## 注意事项

- 生成2048×2048的样本需要较大内存
- 可以根据需要调整image_size参数
- 缺陷信息保存在_info.txt文件中，可用于监督学习训练
