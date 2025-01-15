from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib.pyplot import MultipleLocator
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


#
#
# # --------------------------------------------------------------------
def draw_line2(x, y):
    ax.plot(x[0:4], y[0:4], c="k")
    ax.plot([x[2], x[4]], [y[2], y[4]], c="k")
    ax.plot([x[2], x[11]], [y[2], y[11]], c="k")

    ax.plot(x[4:10], y[4:10], c="k")
    ax.plot([x[7], x[10]], [y[7], y[10]], c="k")

    ax.plot(x[11:17], y[11:17], c="k")
    ax.plot([x[14], x[17]], [y[14], y[17]], c="k")

    ax.plot([x[0], x[22]], [y[0], y[22]], c="k")
    ax.plot([x[0], x[18]], [y[0], y[18]], c="k")

    ax.plot(x[18:22], y[18:22], c="k")
    ax.plot(x[22:26], y[22:26], c="k")

    ax.plot([x[3], x[26], x[27]], [y[3], y[26], y[27]], c="k")
    ax.plot([x[0], x[18]], [y[0], y[18]], c="k")

    # ax.plot([x[27], x[28], x[29]], [y[27], y[28], y[29]], c="k")
    # ax.plot([x[27], x[30], x[31]], [y[27], y[30], y[31]], c="k")


def draw_line3(x, y, z):
    ax.plot3D(x[0:4], y[0:4], z[0:4], c="k")
    ax.plot3D([x[2], x[4]], [y[2], y[4]], [z[2], z[4]], c="k")
    ax.plot3D([x[2], x[11]], [y[2], y[11]], [z[2], z[11]], c="k")

    ax.plot3D(x[4:10], y[4:10], z[4:10], c="k")
    ax.plot3D([x[7], x[10]], [y[7], y[10]], [z[7], z[10]], c="k")

    ax.plot3D(x[11:17], y[11:17], z[11:17], c="k")
    ax.plot3D([x[14], x[17]], [y[14], y[17]], [z[14], z[17]], c="k")

    ax.plot3D([x[0], x[22]], [y[0], y[22]], [z[0], z[22]], c="k")
    ax.plot3D([x[0], x[18]], [y[0], y[18]], [z[0], z[18]], c="k")

    ax.plot3D(x[18:22], y[18:22], z[18:22], c="k")
    ax.plot3D(x[22:26], y[22:26], z[22:26], c="k")

    ax.plot3D([x[3], x[26], x[27]], [y[3], y[26], y[27]], [z[3], z[26], z[27]], c="k")
    ax.plot3D([x[3], x[26]], [y[3], y[26]], [z[3], z[26]], c="k")
    ax.plot3D([x[0], x[18]], [y[0], y[18]], [z[0], z[18]], c="k")

    # ax.plot3D([x[27], x[28], x[29]], [y[27], y[28], y[29]], [z[27], z[28], z[29]], c="k")
    # ax.plot3D([x[27], x[30], x[31]], [y[27], y[30], y[31]], [z[27], z[30], z[31]], c="k")


#
#
# # --------------------------------------------------------------------
#
path = r"D:\\new_data_plus_2\\valid_data\\64.csv"
data = pd.read_csv(path)
x, y, z = [], [], []
for i in range(1, 97, 3):
    x1 = data.iloc[60, i]
    y1 = data.iloc[60, i + 1]
    z1 = data.iloc[60, i + 2]
    x.append(x1)
    y.append(y1)
    z.append(z1)
#
#
# # fig = plt.figure()
# # ax = fig.add_subplot(projection='3d')
# # # 设置三维图图形区域背景颜色（r,g,b,a）
# # ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# # ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# # ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# # # 设置坐标轴不可见
# # ax.axis('off')
# # # ax.set_xlabel('X')
# # # ax.set_ylabel('Y')
# # # ax.set_zlabel('Z')
# # ax.scatter(xs=x, ys=y, zs=z, zdir='z', c="k", marker=".", s=100)
# # draw_line3(x, y, z)
# # # ax.grid(False)
# # # ax.scatter(x=x, y=y, c="k", marker=".", s=100)
# # ax.view_init(elev=111, azim=90)
# # ax.set_xlim(-400, 400)
# # ax.set_xticks([])
# # ax.set_yticks([])
# # ax.set_zticks([])
# #
# # plt.show()
# #
# # # # -------------------------------------------------------------------------------------------------------------
# # 动态图
# cam_df为每一帧的32个节点的分配的0到1的激活值
# path = r"C:\Users\shusheng\Downloads\002.csv"
cam_df = pd.read_csv(path)

# cam_list = []
# for i in range(600):  # CAM获取的帧数
#     cam_list.append(list(cam_df.iloc[i]))
# plt.ion()
# time = 0
raw_path = 'C://Users//shusheng//Desktop//fininal_translation//save//casia_image//'
for time in range(0, 1200, 2):  # 起始帧，结束帧，跳跃帧数
    x, y, z = [], [], []
    for i in range(1, 97, 3):
        x1 = data.iloc[time, i]
        y1 = data.iloc[time, i + 1]
        z1 = data.iloc[time, i + 2]
        x.append(x1)
        y.append(y1)
        z.append(z1)

    plt.clf()  # 清除之前画的图
    fig = plt.gcf()  # 获取当前图
    ax = fig.add_subplot(projection='3d')
    # 设置三维图图形区域背景颜色（r,g,b,a）
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # 设置坐标轴不可见
    ax.axis('off')
    ax.set_xlim(-400, 400)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # 在3D空间中绘制带有颜色的散点
    # ax.scatter(xs=x, ys=y, zs=z, c=color_list, marker="o", s=90)
    ax.scatter(xs=x, ys=y, zs=z, marker="o", s=90)
    draw_line3(x, y, z)
    ax.view_init(elev=111, azim=90)
    plt.pause(0.15)  # 暂停一段时间，控制播放速度，越小越快
    # fname = raw_path + '%d.png' % time
    # plt.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0)    # 保存每一帧画面
    plt.ioff()
plt.show()
