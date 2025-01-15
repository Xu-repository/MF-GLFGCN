"""
手臂与步长之间的相关性系数和显著性水平
pearson系数： 0.3641307103818684
   P-Value： 0.019260235671692456
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats.stats import kendalltau
# （左脚和右脚）所表示1号关节的x轴"7"所在列,要左脚x轴是 21*3+1 = 64,z轴为【66】，同理右脚x轴为25*3+1=76，z轴为【78】
# （左手腕和右手腕）7号关节x轴7*3+1 = 22,z轴【24】,同理右手腕z轴为14*3+1=43，z轴为【45】
from scipy.stats import pearsonr

x = []
y = []
s, e = 0, 600
# 0-41
for i in range(0, 41):
    path1 = "D:\\new_data_plus_2\\train_data\\" + str(i) + ".csv"
    data1 = pd.read_csv(path1)
    hand1 = data1.iloc[s:e:, 42] - data1.iloc[s:e:, 21]
    leg1 = data1.iloc[s:e:, 78] - data1.iloc[s:e:, 66]
    hand_value = max(hand1) - min(hand1)
    leg_value = max(leg1) - min(leg1)
    x.append(int(round(hand_value, 0)))
    y.append(int(round(leg_value, 0)))

r = pearsonr(x, y)
print("pearson系数：", r[0])
print("   P-Value：", r[1])

plt.scatter(x, y)
plt.show()
