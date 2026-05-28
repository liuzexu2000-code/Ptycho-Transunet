import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = 'd:/BaiduSyncdisk/second_term/Uptynet/article_experiment'

def set_chinese_font():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
    plt.rcParams['axes.unicode_minus'] = False

def plot_ptychonn_structure(ax):
    ax.set_title('PtychoNN 网络结构', fontsize=14, pad=20)
    
    layers = [
        ('输入', '1×64×64'),
        ('Conv+ReLU', '64×64×64'),
        ('Conv+ReLU', '64×64×64'),
        ('MaxPool', '64×32×32'),
        ('Conv+ReLU', '128×32×32'),
        ('Conv+ReLU', '128×32×32'),
        ('MaxPool', '128×16×16'),
        ('Conv+ReLU', '256×16×16'),
        ('Conv+ReLU', '256×16×16'),
        ('Upsample', '256×32×32'),
        ('Conv+ReLU', '128×32×32'),
        ('Conv+ReLU', '128×32×32'),
        ('Upsample', '128×64×64'),
        ('Conv+ReLU', '64×64×64'),
        ('输出(振幅)', '1×64×64'),
        ('输出(相位)', '1×64×64'),
    ]
    
    y_pos = range(len(layers))
    layer_names = [f"{name}\n{shape}" for name, shape in layers]
    
    ax.barh(y_pos, [1]*len(layers), height=0.6, color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(layer_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.5)
    ax.set_xticks([])
    ax.grid(axis='x', alpha=0.3)
    
    ax.text(1.2, len(layers)//2, 'U-Net结构\n(双分支解码器)', fontsize=10, 
            ha='center', va='center', bbox=dict(facecolor='lightgreen', alpha=0.5))

def plot_transunet_structure(ax):
    ax.set_title('TransUNet 网络结构', fontsize=14, pad=20)
    
    layers = [
        ('输入', '1×64×64'),
        ('ResidualBlock', '64×64×64'),
        ('MaxPool', '64×32×32'),
        ('ResidualBlock', '128×32×32'),
        ('MaxPool', '128×16×16'),
        ('ResidualBlock', '256×16×16'),
        ('Transformer\nEncoder', '256×16×16'),
        ('Transformer\nBlocks×4', '256×16×16'),
        ('Upsample', '256×32×32'),
        ('Concatenate', '384×32×32'),
        ('ResidualBlock', '128×32×32'),
        ('Upsample', '128×64×64'),
        ('Concatenate', '192×64×64'),
        ('ResidualBlock', '64×64×64'),
        ('输出(振幅)', '1×64×64'),
        ('输出(相位)', '1×64×64'),
    ]
    
    y_pos = range(len(layers))
    layer_names = [f"{name}\n{shape}" for name, shape in layers]
    
    colors = ['skyblue']*5 + ['orange']*3 + ['skyblue']*8
    ax.barh(y_pos, [1]*len(layers), height=0.6, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(layer_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.5)
    ax.set_xticks([])
    ax.grid(axis='x', alpha=0.3)
    
    ax.text(1.2, 6, 'Transformer\n编码器', fontsize=10, 
            ha='center', va='center', bbox=dict(facecolor='orange', alpha=0.5))

def plot_transunet_ptycho_structure(ax):
    ax.set_title('TransUNet-Ptycho 网络结构', fontsize=14, pad=20)
    
    layers = [
        ('输入', '100×64×64'),
        ('ResidualBlock', '64×64×64'),
        ('MaxPool', '64×32×32'),
        ('ResidualBlock', '128×32×32'),
        ('MaxPool', '128×16×16'),
        ('ResidualBlock', '256×16×16'),
        ('Transformer\nEncoder', '256×16×16'),
        ('Transformer\nBlocks×4', '256×16×16'),
        ('Upsample', '256×32×32'),
        ('Concatenate', '384×32×32'),
        ('ResidualBlock', '128×32×32'),
        ('Upsample', '128×64×64'),
        ('Concatenate', '192×64×64'),
        ('ResidualBlock', '64×64×64'),
        ('Upsample', '64×128×128'),
        ('输出(振幅)', '1×128×128'),
        ('输出(相位)', '1×128×128'),
    ]
    
    y_pos = range(len(layers))
    layer_names = [f"{name}\n{shape}" for name, shape in layers]
    
    colors = ['lightcoral'] + ['skyblue']*4 + ['orange']*3 + ['skyblue']*9
    ax.barh(y_pos, [1]*len(layers), height=0.6, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(layer_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.5)
    ax.set_xticks([])
    ax.grid(axis='x', alpha=0.3)
    
    ax.text(1.2, 0, '100张衍射图\n序列输入', fontsize=10, 
            ha='center', va='center', bbox=dict(facecolor='lightcoral', alpha=0.5))
    ax.text(1.2, 6, 'Transformer\n编码器', fontsize=10, 
            ha='center', va='center', bbox=dict(facecolor='orange', alpha=0.5))
    ax.text(1.2, 14, '额外上采样\n至128×128', fontsize=10, 
            ha='center', va='center', bbox=dict(facecolor='lightgreen', alpha=0.5))

def main():
    set_chinese_font()
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    
    plot_ptychonn_structure(axes[0])
    plot_transunet_structure(axes[1])
    plot_transunet_ptycho_structure(axes[2])
    
    plt.tight_layout(pad=3)
    
    output_path = f'{BASE_DIR}/comparison_results/network_structure.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"网络结构图已保存到: {output_path}")
    
    print("\n网络结构总结:")
    print("=" * 80)
    print("\n1. PtychoNN (单一衍射图案重构):")
    print("   - 输入: 1×64×64 (单张衍射图案)")
    print("   - 结构: 传统U-Net，双分支解码器")
    print("   - 输出: 2×64×64 (振幅 + 相位)")
    
    print("\n2. TransUNet (单一衍射图案重构):")
    print("   - 输入: 1×64×64 (单张衍射图案)")
    print("   - 结构: U-Net + Transformer编码器")
    print("   - Transformer Blocks: 4层")
    print("   - 输出: 2×64×64 (振幅 + 相位)")
    
    print("\n3. TransUNet-Ptycho (叠层成像重构):")
    print("   - 输入: 100×64×64 (10×10网格扫描)")
    print("   - 结构: U-Net + Transformer序列建模")
    print("   - Transformer Blocks: 4层")
    print("   - 额外上采样至128×128")
    print("   - 输出: 2×128×128 (完整物体)")

if __name__ == '__main__':
    main()