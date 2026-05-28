import torch
import os

BASE_DIR = 'd:/BaiduSyncdisk/second_term/Uptynet/article_experiment'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 从训练脚本导入模型
import sys
sys.path.insert(0, BASE_DIR + '/scripts')
from train_ptycho_model import TransUNetPtycho

print("测试 TransUNet-Ptycho 模型加载...")
print(f"设备: {device}")

# 创建模型
model = TransUNetPtycho(img_size=64, in_channels=100, embed_dim=256, depth=2, num_heads=8).to(device)
print("模型创建成功")

# 检查权重文件
weight_files = [
    f'{BASE_DIR}/models/best_transunet-ptycho_ptycho.pth',
    f'{BASE_DIR}/models/transunet-ptycho_ptycho_weights.pth'
]

for weight_file in weight_files:
    if os.path.exists(weight_file):
        print(f"找到权重文件: {weight_file}")
        try:
            # 先检查文件内容
            state_dict = torch.load(weight_file, map_location=device)
            print(f"成功读取权重文件，包含 {len(state_dict.keys())} 个参数")
            
            # 打印前5个参数名称
            keys = list(state_dict.keys())[:5]
            print(f"前5个参数: {keys}")
            
            # 尝试加载
            model.load_state_dict(state_dict)
            print("模型加载成功!")
            break
        except Exception as e:
            print(f"加载失败: {str(e)}")
    else:
        print(f"权重文件不存在: {weight_file}")

print("\n测试完成")
