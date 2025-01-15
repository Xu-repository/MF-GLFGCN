"""
对称距离可视化2

27(277.1,562.1)，6(373,815.7)，15(435,903.4)
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# （左脚和右脚）所表示1号关节的x轴"7"所在列,要左脚x轴是 21*3+1 = 64,z轴为【66】，同理右脚x轴为25*3+1=76，z轴为【78】
# （左手腕和右手腕）7号关节x轴7*3+1 = 22,z轴【24】,同理右手腕z轴为14*3+1=43，z轴为【45】,绘制的21是手肘的Z轴，比较直观和清楚

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

plt.figure(figsize=(12, 5))
"""
手臂摆动幅度
"""
sub_1, sub_2, sub_3 = 27, 6, 15
# =================================================================================


s, e = 0, 600
path1 = "D:\\new_data_plus_2\\train_data\\" + str(sub_1) + ".csv"
data1 = pd.read_csv(path1)
data1_l = data1.iloc[s:e:, 21]
data1_r = data1.iloc[s:e:, 42]
data1_sd = data1_r - data1_l
data1_sd = moving_average(data1_sd, window_size=3)
data1_sd = pd.Series(data1_sd.tolist())
print("subject_1", max(data1_sd)-min(data1_sd))  # 33厘米
path2 = "D:\\new_data_plus_2\\train_data\\" + str(sub_2) + ".csv"
data2 = pd.read_csv(path2)
data2_l = data2.iloc[s:e:, 21]
data2_r = data2.iloc[s:e:, 42]
data2_sd = data2_r - data2_l
data2_sd = moving_average(data2_sd, window_size=3)
data2_sd = pd.Series(data2_sd.tolist())
print("subject_2", max(data2_sd)-min(data2_sd))  # 44.8厘米
path3 = "D:\\new_data_plus_2\\train_data\\" + str(sub_3) + ".csv"
data3 = pd.read_csv(path3)
data3_l = data3.iloc[s:e:, 21]
data3_r = data3.iloc[s:e:, 42]
data3_sd = data3_r - data3_l
data3_sd = moving_average(data3_sd, window_size=3)
data3_sd = pd.Series(data3_sd.tolist())
print("subject_3", max(data3_sd)-min(data3_sd))  # 66.4厘米
y_value = [-250, 230]

# 将数据的索引除以30
data1_sd.index = data1_sd.index / 30
data2_sd.index = data2_sd.index / 30
data3_sd.index = data3_sd.index / 30

plt.subplot(3, 2, 1)
ax = data1_sd.plot(color='royalblue', alpha=0.8, linewidth=2)
ax.set_ylim(y_value)
plt.subplot(3, 2, 3)
bx = data2_sd.plot(color='darkorange', alpha=0.8, linewidth=2)
bx.set_ylim(y_value)
bx.set_ylabel("Arm swing distance")

plt.subplot(3, 2, 5)
cx = data3_sd.plot(color='seagreen', alpha=0.8, linewidth=2)
cx.set_ylim(y_value)
cx.set_xlabel("Time(s)\n(a)")

#
# """
# 步长计算
# """
# # =================================================================================
s, e = 0, 600
path1 = "D:\\new_data_plus_2\\train_data\\" + str(sub_1) + ".csv"
data1 = pd.read_csv(path1)
data1_l = data1.iloc[s:e:, 66]
data1_r = data1.iloc[s:e:, 78]
data1_sd = data1_r - data1_l
data1_sd = moving_average(data1_sd, window_size=3)
data1_sd = pd.Series(data1_sd.tolist())
print("subject_1", max(data1_sd)-min(data1_sd))
path2 = "D:\\new_data_plus_2\\train_data\\" + str(sub_2) + ".csv"
data2 = pd.read_csv(path2)
data2_l = data2.iloc[s:e:, 66]
data2_r = data2.iloc[s:e:, 78]
data2_sd = data2_r - data2_l
data2_sd = moving_average(data2_sd, window_size=3)
data2_sd = pd.Series(data2_sd.tolist())
print("subject_2", max(data2_sd)-min(data2_sd))
path3 = "D:\\new_data_plus_2\\train_data\\" + str(sub_3) + ".csv"
data3 = pd.read_csv(path3)
data3_l = data3.iloc[s:e:, 66]
data3_r = data3.iloc[s:e:, 78]
data3_sd = data3_r - data3_l
data3_sd = moving_average(data3_sd, window_size=3)
data3_sd = pd.Series(data3_sd.tolist())
print("subject_3", max(data3_sd)-min(data3_sd))

y_value = [-450, 450]
# 将数据的索引除以30
data1_sd.index = data1_sd.index / 30
data2_sd.index = data2_sd.index / 30
data3_sd.index = data3_sd.index / 30


plt.subplot(3, 2, 2)
ax3 = data1_sd.plot(color='royalblue', alpha=0.8, linewidth=2)
ax3.set_ylim(y_value)
plt.subplot(3, 2, 4)
bx3 = data2_sd.plot(color='darkorange', alpha=0.8, linewidth=2)
bx3.set_ylim(y_value)
bx3.set_ylabel("Step length")

plt.subplot(3, 2, 6)
cx3 = data3_sd.plot(color='seagreen', alpha=0.8, linewidth=2)
cx3.set_ylim(y_value)
cx3.set_xlabel("Time(s)\n(b)")

t_lis = ["Subject 1", "Subject 2", "Subject 3"]
ax.legend([t_lis[0]], loc='upper right')
bx.legend([t_lis[1]], loc='upper right')
cx.legend([t_lis[2]], loc='upper right')

ax3.legend([t_lis[0]], loc='upper right')
bx3.legend([t_lis[1]], loc='upper right')
cx3.legend([t_lis[2]], loc='upper right')

# plt.savefig(r"C:\Users\shusheng\Desktop\论文图片\symmetry3.png", dpi=300, format="png")
plt.show()
