import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Set global font size
plt.rcParams.update({'font.size': 14})

def parse_txt(path):
    with open(path) as f:
        data = np.array([list(map(lambda x: x.strip(), line.split())) for line in f.readlines()])
    
    # Extract numeric part from the first column (assuming it's the epoch information)
    data[:, 0] = [int(epoch.split('/')[0]) for epoch in data[:, 0]]
    
    return data

# 模型名称
names = ['Baseline', 'Small', 'BiFPN', 'DyHead', 'BiFPN + DyHead', 'Small + DyHead', 'Small + BiFPN', 'Proposed Model']

# 创建一个更大的图形
plt.figure(figsize=(12, 5))

# 第一个子图：mAP_0.5
plt.subplot(1, 2, 1)
# 设置纵坐标轴范围从0.35到0.56，每0.05取一次
plt.yticks(np.arange(0.35, 0.57, 0.05))

for model_name in names:
    data = parse_txt(f'runs/train/{model_name}/results.txt')
    epochs = np.array(data[:, 0], dtype=int)
    mAP_05 = np.array(data[:, 10], dtype=float)
    
    # 使用三次样条插值
    x_smooth = np.linspace(epochs.min(), epochs.max(), 300)
    y_smooth = make_interp_spline(epochs, mAP_05)(x_smooth)
    
    plt.plot(x_smooth, y_smooth, label=model_name, linestyle='-', linewidth=2)

plt.xlabel('Epoch')
plt.title(r'${mAP}_{0.5}$') 
plt.legend()

# Remove grid lines
plt.grid(False)

# 设置横坐标轴刻度，每100轮显示一个
plt.xticks(np.arange(0, max(epochs)+1, step=100))

# 第二个子图：mAP_0.5:0.95
plt.subplot(1, 2, 2)
# 设置纵坐标轴范围从0.2到0.33，每0.05取一次
plt.yticks(np.arange(0.2, 0.34, 0.05))

for model_name in names:
    data = parse_txt(f'runs/train/{model_name}/results.txt')
    epochs = np.array(data[:, 0], dtype=int)
    mAP_095 = np.array(data[:, 11], dtype=float)
    
    # 使用三次样条插值
    x_smooth = np.linspace(epochs.min(), epochs.max(), 300)
    y_smooth = make_interp_spline(epochs, mAP_095)(x_smooth)
    
    plt.plot(x_smooth, y_smooth, label=model_name, linestyle='-', linewidth=2)

plt.xlabel('Epoch')
plt.title(r'${mAP}_{0.95}$') 
plt.legend()

# Remove grid lines
plt.grid(False)

# 设置横坐标轴刻度，每100轮显示一个
plt.xticks(np.arange(0, max(epochs)+1, step=100))

# 调整布局以防止重叠
plt.tight_layout()

# 保存图形
plt.savefig('smoothed_academic_curve.pdf')

# 显示图形
plt.show()
