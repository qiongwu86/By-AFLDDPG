import numpy as np
# AFL_weight vs AFL : acc_bad node 对比
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

names = ['1', '2', '3', '4', '5', '6', '7','8','9','10']
x = range(len(names))
# x = range(,10)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字


y_14 = [0.8412, 0.9219, 0.9398, 0.9465, 0.9528, 0.9543, 0.9578, 0.9579, 0.9634, 0.9642]  # AFL_weight
y_24 = [0.6971, 0.7932, 0.9008, 0.9286, 0.9268, 0.121, 0.0974, 0.0974, 0.0974, 0.6826] # AFL
y_34 = [0.1090, 0.2327, 0.3794, 0.3445, 0.5717, 0.7295, 0.7535, 0.7728, 0.8059, 0.7764] # FL
y_341 = [0.0871, 0.0778, 0.5219, 0.7320, 0.5941, 0.7547, 0.8398, 0.8626, 0.7905, 0.8508] # FL acc2
y_3411 = [0.0772, 0.0861, 0.1659, 0.6048, 0.4298, 0.7412, 0.8047, 0.3231, 0.739, 0.7974]
# plt.plot(x, y_14, color='orangered', marker='o', linestyle='-', label='本文方案')
# plt.plot(x, y_24, color='blueviolet', marker='D', linestyle='-.', label='传统异步联邦方案')
# plt.plot(x, y_3411, color='green', marker='*', linestyle=':', label='传统联邦方案')

plt.plot(x, y_14, color='orangered', marker='o', linestyle='-', label='our scheme')
plt.plot(x, y_24, color='blueviolet', marker='D', linestyle='-.', label='traditional AFL')
plt.plot(x, y_3411, color='green', marker='*', linestyle=':', label='traditional FL')

plt.legend()  # 显示图例
# plt.xticks(x, names, rotation=45)
# plt.xticks(x, names)
plt.xticks(x, names)
# plt.xlabel("步数")  # X轴标签
# plt.ylabel("精度")  # Y轴标签
plt.xlabel("number of step")  # X轴标签
plt.ylabel("accuracy")  # Y轴标签

plt.show()


# loss:
y_1 = [1.7168, 0.8660, 0.6458, 0.5460, 0.4765, 0.4429, 0.4130, 0.3918, 0.3667, 0.3423]  # AFL_weight
y_2 = [1.7509, 1.0709, 1.2095, 0.7316, 0.6874, 0.8635, 2.1154, 2.3044, 2.3032, 2.1857]  # AFL
y_3 = [2.1778, 2.2197, 1.9896, 1.8918, 1.8352, 1.6910, 1.6494, 1.6570, 1.8329, 1.8548]  # FL
y_31 = [5.0825, 2.0824, 2.1407, 2.4323, 1.5807, 1.5878,1.5098, 1.4159, 1.3202, 1.3431]  # FL loss2
y_311 = [2.2044, 2.3041, 2.3085, 2.8498, 3.0494, 1.8160, 1.8043, 3.3801, 1.6547, 1.4262]
# plt.plot(x, y_1, color='orangered', marker='o', linestyle='-', label='本文方案')
# plt.plot(x, y_2, color='blueviolet', marker='D', linestyle='-.', label='传统异步联邦方案')
# plt.plot(x, y_311, color='green', marker='*', linestyle=':', label='传统联邦方案')

plt.plot(x, y_1, color='orangered', marker='o', linestyle='-', label='our scheme')
plt.plot(x, y_2, color='blueviolet', marker='D', linestyle='-.', label='traditional AFL')
plt.plot(x, y_311, color='green', marker='*', linestyle=':', label='traditional FL')

plt.legend()  # 显示图例
# plt.xticks(x, names, rotation=45)
# plt.xticks(x, names)
plt.xticks(x, names)
# plt.xlabel("步数")  # X轴标签
# plt.ylabel("损失")  # Y轴标签
plt.xlabel("number of step")  # X轴标签
plt.ylabel("loss")  # Y轴标签

plt.show()

# # AFL:
# [1.7509, 1.0709, 1.2095, 0.7316, 0.6874, 0.8635, 2.1154, 2.3044, 2.3032, 2.1857]
# [0.6971, 0.7932, 0.9008, 0.9286, 0.9268, 0.121, 0.0974, 0.0974, 0.0974, 0.6826]
#
#
# # FL:
# [2.1778, 2.2197, 1.9896, 1.8918, 1.8352, 1.6910, 1.6494, 1.6570, 1.8329, 1.8548]
# tracc is: [0.1090, 0.2327, 0.3794, 0.3445, 0.5717, 0.7295, 0.7535, 0.7728, 0.8059, 0.7764]
#
# [5.0825, 2.0824, 2.1407, 2.4323, 1.5807, 1.5878,1.5098, 1.4159, 1.3202, 1.3431]
# tracc is: [0.0871, 0.0778, 0.5219, 0.7320, 0.5941, 0.7547, 0.8398, 0.8626, 0.7905, 0.8508]
[2.2044, 2.3041, 2.3085, 2.8498, 3.0494, 1.8160, 1.8043, 8.3801, 1.6547, 1.4262]
[0.0772, 0.0861, 0.1659, 0.6048, 0.4298, 0.7412, 0.8047, 0.3231, 0.739, 0.7974]
