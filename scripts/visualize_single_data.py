import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = 'd:/BaiduSyncdisk/second_term/Uptynet/article_experiment'

def visualize_data():
    data_dir = f'{BASE_DIR}/data_single/test'
    input_dir = os.path.join(data_dir, 'inputs')
    label_dir = os.path.join(data_dir, 'labels')
    
    input_files = sorted([f for f in os.listdir(input_dir) if f.startswith('input_')])
    
    num_samples = min(5, len(input_files))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    for i in range(num_samples):
        input_path = os.path.join(input_dir, input_files[i])
        base_name = input_files[i].replace('input_', '')
        label_path = os.path.join(label_dir, f'label_{base_name}')
        
        diffraction = np.load(input_path)
        label = np.load(label_path)
        
        amp_label = label[0]
        phase_label = label[1]
        
        axes[i, 0].imshow(diffraction, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Diffraction Pattern {i+1}', fontsize=12)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(amp_label, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Amplitude Label {i+1}', fontsize=12)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(phase_label, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Phase Label {i+1}', fontsize=12)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    output_path = f'{BASE_DIR}/comparison_results/single_data_samples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"可视化结果已保存到: {output_path}")
    
    print("\n数据统计信息:")
    print(f"测试样本数量: {len(input_files)}")
    print(f"衍射图案尺寸: {diffraction.shape}")
    print(f"标签尺寸: {label.shape}")
    print(f"衍射图案范围: [{diffraction.min():.4f}, {diffraction.max():.4f}]")
    print(f"振幅标签范围: [{amp_label.min():.4f}, {amp_label.max():.4f}]")
    print(f"相位标签范围: [{phase_label.min():.4f}, {phase_label.max():.4f}]")

if __name__ == '__main__':
    visualize_data()