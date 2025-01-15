"""
数据可视化

27(277.1,562.1)，6(373,815.7)，15(435,903.4)
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# （左脚和右脚）所表示1号关节的x轴"7"所在列,要左脚x轴是 21*3+1 = 64,z轴为【66】，同理右脚x轴为25*3+1=76，z轴为【78】
# （左手腕和右手腕）7号关节x轴7*3+1 = 22,z轴【24】,同理右手腕z轴为14*3+1=43，z轴为【45】,绘制的21是手肘的Z轴，比较直观和清楚

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

plt.figure(figsize=(12, 5),facecolor='None')
"""
手臂摆动幅度
"""
sub_1, sub_2, sub_3 = 27, 6, 15
# =================================================================================

s, e = 0, 600
# path1 = "D:\\new_data_plus_2\\train_data\\" + str(sub_1) + ".csv"
path1 = r"D:\22届采集的数据\Kinect数据\3.严云飞-右脚\跑步机\正常步态.csv"

data1 = pd.read_csv(path1)
data1_x = data1.iloc[s:e, 141]
data1_y = data1.iloc[s:e, 142]
data1_z = data1.iloc[s:e, 143]

# data1_sd = moving_average(data1_x, window_size=4)
# data1_x = pd.Series(data1_sd.tolist())
#
# data1_sd = moving_average(data1_y, window_size=4)
# data1_y = pd.Series(data1_sd.tolist())
#
# data1_sd = moving_average(data1_z, window_size=4)
# data1_z = pd.Series(data1_sd.tolist())

# 将数据的索引除以30
data1_x.index = data1_x.index / 30
data1_y.index = data1_y.index / 30
data1_z.index = data1_z.index / 30


plt.subplot(3, 1, 1)
ax = data1_x.plot(color='royalblue', alpha=0.8, linewidth=2)
ax.legend(["x-axis"], loc='upper right')

plt.subplot(3, 1, 2)
bx = data1_y.plot(color='darkorange', alpha=0.8, linewidth=2)
bx.legend(["y-axis"], loc='upper right')
bx.set_ylabel("Value(mm)")

plt.subplot(3, 1, 3)
cx = data1_z.plot(color='seagreen', alpha=0.8, linewidth=2)
cx.legend(["z-axis"], loc='upper right')
cx.set_xlabel("Time(s)")

# t_lis = ["Subject 1", "Subject 2", "Subject 3"]
# ax.legend([t_lis[0]], loc='upper right')
# bx.legend([t_lis[1]], loc='upper right')
# cx.legend([t_lis[2]], loc='upper right')
#
# ax3.legend([t_lis[0]], loc='upper right')
# bx3.legend([t_lis[1]], loc='upper right')
# cx3.legend([t_lis[2]], loc='upper right')

# plt.savefig(r"C:\Users\shusheng\Desktop\论文图片\symmetry3.png", dpi=300, format="png")
plt.show()
