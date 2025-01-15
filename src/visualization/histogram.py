# 柱状图绘制

import matplotlib.pyplot as plt
import numpy as np

# 数据
# x_label = ["1_1_Conv", "Linear", "Max", "Median", "Mean"]
# y1_label = [91.18, 95.59, 94.41, 95, 97.06]
# y2_label = [92.65, 95.59, 97.06, 95.59, 98.53]

x_label = ["1_1_Conv_2", "Linear_2", "Cat", "Sum"]
y1_label = [89.71, 91.77, 95.59, 97.06]
y2_label = [95.59, 95.59, 95.59, 98.53]

# 设置x轴数据
x = np.arange(len(x_label))
# width = 0.35
width = 0.3
y1_x = x
y2_x = x + width

# 设置绘图风格
plt.style.use('seaborn-whitegrid')

# 创建柱状图
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(y1_x, y1_label, width=width, align="center", color="teal", label="Mean Accuracy")
bars2 = ax.bar(y2_x, y2_label, width=width, align="center", color="darkorange", label="Max Accuracy")

# 设置x轴标签
ax.set_xticks(x + width / 2)
ax.set_xticklabels(x_label, fontsize=12, weight='bold')

# 添加数据标签
for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.2f}%', va="bottom", ha="center", fontsize=11)

for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.2f}%', va="bottom", ha="center", fontsize=11)

# 设置图例
ax.legend(loc="upper left", fontsize=12)

# 设置y轴标签
ax.set_ylabel('Accuracy (%)', fontsize=14, weight='bold')

# 设置y轴的起始值为60
ax.set_ylim(70, 100)

# 设置网格线和背景
ax.grid(True, linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()
