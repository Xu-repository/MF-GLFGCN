"""
x,y,z轴数据可视化
陈智睿*
严云飞*
林晓露*
黄娜*
杨亨铄*
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

c1 = 'gold'
c2 = 'orangered'
c3 = 'deepskyblue'
a, b = 0, 1  # 颜色透明度0.3,0.85
a1, b1 = 2, 2  # 线条粗细2,2.5
x_num, y_num = 1, 1

plt.figure(figsize=(12, 4),facecolor='None')

# 定义移动平均滤波函数
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# # 读取数据

#
# #
# path5 = r"D:\原始数据\后\陈智睿\1正常.csv"
# data5 = pd.read_csv(path5)

path = r"D:\22届采集的数据\Kinect数据\3.严云飞-右脚\跑步机\正常步态.csv"
data = pd.read_csv(path)

# path4 = r"D:\22届采集的数据\Kinect数据\14.杨亨铄-右脚\跑步机\正常步态.csv"
# data4 = pd.read_csv(path4)



data1_z = data.iloc[0:600, 171]  # 右脚踝z轴数据
data1_f_z = moving_average(data1_z, window_size=8)

# data2_z = data.iloc[0:600, 143]     # 左脚踝z轴数据
data2_z = data.iloc[12:612, 143]  # 左踝关节前移13帧
data2_f_z = moving_average(data2_z, window_size=8)

plt.subplot(x_num, y_num, 1)
x = np.arange(0, 593)
# plt.plot(np.arange(0, 600), data1_z, linestyle='-', color=c3, alpha=a, linewidth=a1, label='Original Data')
plt.plot(x/30, data1_f_z, color=c3, alpha=b, linewidth=b1, label='Left ankle')
# plt.plot(x/30, data2_f_z, color=c1, alpha=b, linewidth=b1, label='Right ankle')
plt.xlabel('Time(s)', fontsize=12)
plt.ylabel('Distance(millimetre)', fontsize=12)
# # 显示图例，位置在右上角
plt.legend(loc='upper right', fontsize=12)
# MAE = sum(abs(data1_f_z - data2_f_z)) / len(x)
# print("MAE:", MAE)
# 使用紧凑的布局
plt.tight_layout()

# 显示图表
plt.show()
