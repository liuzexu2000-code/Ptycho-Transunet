import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = 'd:/BaiduSyncdisk/second_term/Uptynet/article_experiment'

def visualize_ptycho_data():
    data_dir = f'{BASE_DIR}/data_ptycho/test'
    input_dir = os.path.join(data_dir, 'inputs')
    label_dir = os.path.join(data_dir, 'labels')
    
    input_files = sorted([f for f in os.listdir(input_dir) if f.startswith('input_')])
    
    if not input_files:
        print("警告: 测试集为空!")
        return
    
    num_samples = min(3, len(input_files))
    
    for sample_idx in range(num_samples):
        input_path = os.path.join(input_dir, input_files[sample_idx])
        base_name = input_files[sample_idx].replace('input_', '')
        label_path = os.path.join(label_dir, f'label_{base_name}')
        
        diffraction_stack = np.load(input_path)
        label = np.load(label_path)
        
        amp_label = label[0]
        phase_label = label[1]
        
        num_diffraction = diffraction_stack.shape[0]
        grid_size = int(np.sqrt(num_diffraction))
        patch_size = diffraction_stack.shape[1]
        full_size = amp_label.shape[0]
        
        fig, axes = plt.subplots(6, 6, figsize=(18, 18))
        
        for i in range(min(25, num_diffraction)):
            row = i // 5
            col = i % 5
            axes[row, col].imshow(diffraction_stack[i], cmap='gray', vmin=0, vmax=1)
            axes[row, col].set_title(f'Diff {i+1}', fontsize=8)
            axes[row, col].axis('off')
        
        for i in range(5, 6):
            for j in range(6):
                axes[i, j].axis('off')
        
        axes[5, 0].imshow(amp_label, cmap='gray', vmin=0, vmax=1)
        axes[5, 0].set_title(f'Amplitude ({full_size}×{full_size})', fontsize=10)
        axes[5, 0].axis('off')
        
        axes[5, 1].imshow(phase_label, cmap='gray', vmin=0, vmax=1)
        axes[5, 1].set_title(f'Phase ({full_size}×{full_size})', fontsize=10)
        axes[5, 1].axis('off')
        
        center_idx = grid_size * grid_size // 2
        axes[5, 2].imshow(diffraction_stack[center_idx], cmap='gray', vmin=0, vmax=1)
        axes[5, 2].set_title(f'Center Diffraction ({patch_size}×{patch_size})', fontsize=10)
        axes[5, 2].axis('off')
        
        plt.suptitle(f'Ptycho Dataset Sample {sample_idx+1}', fontsize=14)
        plt.tight_layout()
        
        output_path = f'{BASE_DIR}/comparison_results/ptycho_data_sample_{sample_idx+1}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"可视化结果已保存到: {output_path}")
    
    print("\n数据统计信息:")
    print(f"测试样本数量: {len(input_files)}")
    print(f"衍射图案尺寸: {diffraction_stack.shape}")
    print(f"标签尺寸: {label.shape}")
    print(f"衍射图案数量: {num_diffraction} ({grid_size}×{grid_size})")
    print(f"每张衍射图尺寸: {patch_size}×{patch_size}")
    print(f"完整物体尺寸: {full_size}×{full_size}")
    print(f"衍射图案范围: [{diffraction_stack.min():.4f}, {diffraction_stack.max():.4f}]")
    print(f"振幅标签范围: [{amp_label.min():.4f}, {amp_label.max():.4f}]")
    print(f"相位标签范围: [{phase_label.min():.4f}, {phase_label.max():.4f}]")

if __name__ == '__main__':
    visualize_ptycho_data()