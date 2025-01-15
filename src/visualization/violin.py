import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# 定义关节名称的映射字典
joint_names = {
    0: "pelvis", 1: "spine_naval", 2: "spine_chest", 3: "neck", 4: "clavicle_left",
    5: "shoulder_left", 6: "elbow_left", 7: "wrist_left", 8: "hand_left", 9: "handtip_left",
    10: "thumb_left", 11: "clavicle_right", 12: "shoulder_right", 13: "elbow_right",
    14: "wrist_right", 15: "hand_right", 16: "handtip_right", 17: "thumb_right",
    18: "hip_left", 19: "knee_left", 20: "ankle_left", 21: "foot_left", 22: "hip_right",
    23: "knee_right", 24: "ankle_right", 25: "foot_right", 26: "head", 27: "nose",
    28: "eye_left", 29: "ear_left", 30: "eye_right", 31: "ear_right"
}
# sns.palplot(sns.color_palette("Accent"))
# 读取数据
path1 = r"D:\22届采集的数据\Kinect数据\1.王亮-右脚\跑步机\正常步态.csv"
data1 = pd.read_csv(path1, index_col=0).iloc[:, 2::7]
data1.columns = [joint_names[i] for i in range(data1.shape[1])]

path2 = r"D:\22届采集的数据\Kinect数据\6.苏良宽-右脚\跑步机\正常步态.csv"
data2 = pd.read_csv(path2, index_col=0).iloc[:, 2::7]
data2.columns = [joint_names[i] for i in range(data2.shape[1])]

path3 = r"D:\22届采集的数据\Kinect数据\30.黄杰胜-左脚\跑步机\正常步态.csv"
data3 = pd.read_csv(path3, index_col=0).iloc[:, 2::7]
data3.columns = [joint_names[i] for i in range(data3.shape[1])]

# 设置绘图风格为适合论文的风格
sns.set(style="darkgrid", font_scale=1.2, rc={"lines.linewidth": 1.5})

custom_colors = ["Gold", "Gold", "Gold", "Gold", '#FF8C75',"#FF8C75","#FF8C75","#FF8C75","#FF8C75","#FF8C75","#FF8C75","#76D7C4","#76D7C4","#76D7C4","#76D7C4","#76D7C4","#76D7C4","#76D7C4","#8FAADC","#8FAADC","#8FAADC","#8FAADC","#80d6ff","#80d6ff","#80d6ff","#80d6ff","#fcb1b1","#fcb1b1","#fcb1b1","#fcb1b1","#fcb1b1","#fcb1b1"]

# 创建一个包含三个子图的纵向图表
fig, axes = plt.subplots(3, 1, figsize=(13, 18), sharex=True)

# 绘制第一个小提琴图
sns.violinplot(ax=axes[0], data=data1, inner="box", linewidth=1.2,palette=custom_colors)
axes[0].set_ylabel('Amplitude')

# 绘制第二个小提琴图
sns.violinplot(ax=axes[1], data=data2, inner="box", linewidth=1.2,palette=custom_colors)
axes[1].set_ylabel('Amplitude')

# 绘制第三个小提琴图
sns.violinplot(ax=axes[2], data=data3, inner="box", linewidth=1.2,palette=custom_colors)
axes[2].set_ylabel('Amplitude')

# 设置x轴为关节名称，并添加总标题
plt.xlabel('Joint')
plt.xticks(rotation=90)
# 调整布局以避免重叠
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# 设置tight bbox
# fig.savefig('violin.png')
# 显示图表
plt.show()
