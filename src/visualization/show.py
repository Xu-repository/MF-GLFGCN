"""
对称距离可视化

27(277.1,562.1)，6(373,815.7)，15(435,903.4)
"""
import matplotlib.pyplot as plt
import pandas as pd

# （左脚和右脚）所表示1号关节的x轴"7"所在列,要左脚x轴是 21*3+1 = 64,z轴为【66】，同理右脚x轴为25*3+1=76，z轴为【78】
# （左手腕和右手腕）7号关节x轴7*3+1 = 22,z轴【24】,同理右手腕z轴为14*3+1=43，z轴为【45】,绘制的21是手肘的Z轴，比较直观和清楚

plt.figure(figsize=(15, 12))
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
print("subject_1", max(data1_sd), min(data1_sd))  # 33厘米
path2 = "D:\\new_data_plus_2\\train_data\\" + str(sub_2) + ".csv"
data2 = pd.read_csv(path2)
data2_l = data2.iloc[s:e:, 21]
data2_r = data2.iloc[s:e:, 42]
data2_sd = data2_r - data2_l
print("subject_2", max(data2_sd), min(data2_sd))  # 44.8厘米
path3 = "D:\\new_data_plus_2\\train_data\\" + str(sub_3) + ".csv"
data3 = pd.read_csv(path3)
data3_l = data3.iloc[s:e:, 21]
data3_r = data3.iloc[s:e:, 42]
data3_sd = data3_r - data3_l
print("subject_3", max(data3_sd), min(data3_sd))  # 66.4厘米
y_value = [-250, 230]

plt.subplot(6, 2, 1)
ax = data1_sd.plot(color='royalblue', alpha=0.8, linewidth=2)
ax.set_ylim(y_value)
plt.subplot(6, 2, 3)
bx = data2_sd.plot(color='darkorange', alpha=0.8, linewidth=2)
bx.set_ylim(y_value)
plt.subplot(6, 2, 5)
cx = data3_sd.plot(color='seagreen', alpha=0.8, linewidth=2)
cx.set_ylim(y_value)
#
# # =================================================================================
s, e = 0, 2000
path1 = "D:\\new_data_plus_2\\train_data\\" + str(sub_1) + ".csv"
data1 = pd.read_csv(path1)
data1_l = data1.iloc[s:e:, 21]
data1_r = data1.iloc[s:e:, 42]
data1_sd = data1_r - data1_l
# print("subject_1", max(data1_sd), min(data1_sd))
path2 = "D:\\new_data_plus_2\\train_data\\" + str(sub_2) + ".csv"
data2 = pd.read_csv(path2)
data2_l = data2.iloc[s:e:, 21]
data2_r = data2.iloc[s:e:, 42]
data2_sd = data2_r - data2_l
# print("subject_2", max(data2_sd), min(data2_sd))
path3 = "D:\\new_data_plus_2\\train_data\\" + str(sub_3) + ".csv"
data3 = pd.read_csv(path3)
data3_l = data3.iloc[s:e:, 21]
data3_r = data3.iloc[s:e:, 42]
data3_sd = data3_r - data3_l
# print("subject_3", max(data3_sd), min(data3_sd))

plt.subplot(6, 2, 2)
ax2 = data1_sd.plot(color='royalblue', alpha=0.8, linewidth=2)
ax2.set_ylim(y_value)
plt.subplot(6, 2, 4)
bx2 = data2_sd.plot(color='darkorange', alpha=0.8, linewidth=2)
bx2.set_ylim(y_value)
plt.subplot(6, 2, 6)
cx2 = data3_sd.plot(color='seagreen', alpha=0.8, linewidth=2)
cx2.set_ylim(y_value)

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
print("subject_1", max(data1_sd), min(data1_sd))
path2 = "D:\\new_data_plus_2\\train_data\\" + str(sub_2) + ".csv"
data2 = pd.read_csv(path2)
data2_l = data2.iloc[s:e:, 66]
data2_r = data2.iloc[s:e:, 78]
data2_sd = data2_r - data2_l
print("subject_2", max(data2_sd), min(data2_sd))
path3 = "D:\\new_data_plus_2\\train_data\\" + str(sub_3) + ".csv"
data3 = pd.read_csv(path3)
data3_l = data3.iloc[s:e:, 66]
data3_r = data3.iloc[s:e:, 78]
data3_sd = data3_r - data3_l
print("subject_3", max(data3_sd), min(data3_sd))

y_value = [-450, 450]

plt.subplot(6, 2, 7)
ax3 = data1_sd.plot(color='royalblue', alpha=0.8, linewidth=2)
ax3.set_ylim(y_value)
plt.subplot(6, 2, 9)
bx3 = data2_sd.plot(color='darkorange', alpha=0.8, linewidth=2)
bx3.set_ylim(y_value)
plt.subplot(6, 2, 11)
cx3 = data3_sd.plot(color='seagreen', alpha=0.8, linewidth=2)
cx3.set_ylim(y_value)
cx3.set_xlabel("Frame")
#
# # =================================================================================
s, e = 0, 2000
path1 = "D:\\new_data_plus_2\\train_data\\" + str(sub_1) + ".csv"
data1 = pd.read_csv(path1)
data1_l = data1.iloc[s:e:, 66]
data1_r = data1.iloc[s:e:, 78]
data1_sd = data1_r - data1_l
# print("subject_1", max(data1_sd), min(data1_sd))
path2 = "D:\\new_data_plus_2\\train_data\\" + str(sub_2) + ".csv"
data2 = pd.read_csv(path2)
data2_l = data2.iloc[s:e:, 78]
data2_r = data2.iloc[s:e:, 66]
data2_sd = data2_r - data2_l
# print("subject_2", max(data2_sd), min(data2_sd))
path3 = "D:\\new_data_plus_2\\train_data\\" + str(sub_3) + ".csv"
data3 = pd.read_csv(path3)
data3_l = data3.iloc[s:e:, 78]
data3_r = data3.iloc[s:e:, 66]
data3_sd = data3_r - data3_l
# print("subject_3", max(data3_sd), min(data3_sd))


plt.subplot(6, 2, 8)
ax4 = data1_sd.plot(color='royalblue', alpha=0.8, linewidth=2)
ax4.set_ylim(y_value)
plt.subplot(6, 2, 10)
bx4 = data2_sd.plot(color='darkorange', alpha=0.8, linewidth=2)
bx4.set_ylim(y_value)
plt.subplot(6, 2, 12)
cx4 = data3_sd.plot(color='seagreen', alpha=0.8, linewidth=2)
cx4.set_ylim(y_value)
cx4.set_xlabel("Frame")

t_lis = ["Subject 1", "Subject 2", "Subject 3"]
ax.legend([t_lis[0]], loc='upper right')
bx.legend([t_lis[1]], loc='upper right')
cx.legend([t_lis[2]], loc='upper right')

ax2.legend([t_lis[0]], loc='upper right')
bx2.legend([t_lis[1]], loc='upper right')
cx2.legend([t_lis[2]], loc='upper right')

ax3.legend([t_lis[0]], loc='upper right')
bx3.legend([t_lis[1]], loc='upper right')
cx3.legend([t_lis[2]], loc='upper right')

ax4.legend([t_lis[0]], loc='upper right')
bx4.legend([t_lis[1]], loc='upper right')
cx4.legend([t_lis[2]], loc='upper right')
# plt.savefig("symmetry.png", dpi=300, format="png")
plt.show()
