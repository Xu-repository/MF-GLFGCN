import matplotlib.pyplot as plt
import numpy as np

# Sample data
gallery_angles = np.array([0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180])
nm_w_o_pmr = [0.5842, 0.5500, 0.4900, 0.4200, 0.3600, 0.3200, 0.3600, 0.4200, 0.4900, 0.5500, 0.5842]
nm_w_pmr = [0.6224, 0.5900, 0.5400, 0.4800, 0.4300, 0.4000, 0.4300, 0.4800, 0.5400, 0.5900, 0.6224]
bg_w_o_pmr = [0.4011, 0.3700, 0.3300, 0.2800, 0.2300, 0.2100, 0.2300, 0.2800, 0.3300, 0.3700, 0.4011]
bg_w_pmr = [0.4810, 0.4500, 0.4100, 0.3700, 0.3300, 0.3000, 0.3300, 0.3700, 0.4100, 0.4500, 0.4810]
cl_w_o_pmr = [0.3334, 0.3000, 0.2600, 0.2100, 0.1800, 0.1600, 0.1800, 0.2100, 0.2600, 0.3000, 0.3334]
cl_w_pmr = [0.4121, 0.3800, 0.3400, 0.3000, 0.2600, 0.2300, 0.2600, 0.3000, 0.3400, 0.3800, 0.4121]
probe_angles = [0, 90, 180]
# data = [
#     [nm_w_o_pmr, nm_w_pmr, bg_w_o_pmr, bg_w_pmr, cl_w_o_pmr, cl_w_pmr],
#     [list(reversed(nm_w_o_pmr)), list(reversed(nm_w_pmr)), list(reversed(bg_w_o_pmr)), list(reversed(bg_w_pmr)),
#      list(reversed(cl_w_o_pmr)), list(reversed(cl_w_pmr))],
#     [nm_w_o_pmr, nm_w_pmr, bg_w_o_pmr, bg_w_pmr, cl_w_o_pmr, cl_w_pmr]
# ]

labels = ['Mean', 'Max']
markers = ['*', 'o', '^', 'v', '<', '>']
x_label = ["Temporal windows size", "Channel Reduction ratio", "Max Graph Distance"]

t_x = [3, 5, 7, 9]
t_y1 = [0.9383, 0.9706, 0.9471, 0.953]
t_y2 = [0.9559, 0.9706, 0.9559, 0.9559]
c_x = [2, 4, 6, 8]
c_y1 = [0.9706, 0.9706, 0.9294, 0.95]
c_y2 = [0.9706, 0.9706, 0.9412, 0.9706]
m_x = [1, 2, 3, 4]
m_y1 = [0.9706, 0.9324, 0.9412, 0.9383]
m_y2 = [0.9706, 0.9559, 0.9706, 0.9706]
x_value = [t_x, c_x, m_x]
data = [[t_y1, t_y2], [c_y1, c_y2], [m_y1, m_y2]]
y_value = [92, 93, 94, 95, 96, 97, 98, 99, 1]
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i in range(3):
    for j in range(2):
        axs[i].plot(x_value[i], data[i][j], marker=markers[j],
                    label=f"{labels[j]}")

    axs[i].set_xlabel(x_label[i])
    # axs[i].set_title(f'Probe view angle: {probe_angles[i]}°')
    axs[i].set_xticks(x_value[i])
    # axs[i].set_xticks(y_value[i])
    axs[i].legend(loc='best')
    axs[i].grid(True)
axs[0].set_ylabel('Identification Accuracy (%)')

plt.tight_layout()
plt.show()
