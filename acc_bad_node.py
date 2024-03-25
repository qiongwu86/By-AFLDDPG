import numpy as np
# AFL_weight vs AFL : acc_bad node 对比
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

names = ['1', '2', '3', '4', '5', '6', '7','8','9','10']
x = range(len(names))
# x = range(,10)
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
# #data flip
# y1=[0.8566, 0.9089, 0.9261, 0.9354, 0.9459, 0.9538, 0.9548, 0.9565, 0.9574, 0.9588]#our scheme
# y2=[0.8292, 0.8488, 0.8743, 0.8833, 0.9061, 0.9137, 0.922, 0.917, 0.9287, 0.9326]#
# l1=[2.058574808429171, 1.0373395917923365, 0.724040201213565, 0.5926932017800194, 0.5193072918573339, 0.461481252815159, 0.40404763435534824, 0.38091036063163736, 0.3800226466953776, 0.32598931271180603]
# l2=[2.018649822954801, 1.5109132659659863, 1.3553277886161523, 1.263972158496042, 1.2424109577005862, 1.1966955143196742, 1.1314321340753326, 1.138399085289964, 1.1197362017235968, 1.0914623811770774]
# y11=[0.8004, 0.8886, 0.9085, 0.9226, 0.9342, 0.9414, 0.945, 0.9495, 0.9539, 0.9569]
# y22=[0.7214, 0.7837, 0.8371, 0.8492, 0.8901, 0.8873, 0.9188, 0.921, 0.929, 0.9384]
# l11=[2.0537871342978846, 1.1048026597213814, 0.8357963057913717, 0.6871213365207871, 0.5869683208570368, 0.540873555609235, 0.4776993532787131, 0.4637080110833615, 0.41709349198625506, 0.3679480244709139]
# l22=[2.1034036114649295, 1.9860615064831615, 1.8447934330424978, 1.8572553974172499, 1.8996438008857814, 1.7584784802957946, 1.9137608216768878, 1.6275301559743191, 1.624863011889465, 1.7850170200945742]
# plt.plot(x, l11, color='orangered', marker='o', linestyle='-', label='our scheme')
# plt.plot(x, l22, color='blueviolet', marker='D', linestyle='-.', label='without considering bzt')
# plt.legend()  # 显示图例
# plt.xlabel("number of step")  # X轴标签
# plt.ylabel("loss")  # Y轴标签
# plt.show()
# names = ['0', '0.2', '0.4', '0.6', '0.8']
# x=names
# y1=[0.037,0.041,0.045,0.048 ,0.051]
# y2=[0.048,0.082,0.109,0.343,0.892]
# y3=[0.040,0.042,0.044,0.044,0.045]
# y4=[0.048,0.083,0.123,0.411,0.901]
# plt.plot(x, y3, color='orangered', marker='o', linestyle='-', label='our scheme')
# plt.plot(x, y4, color='blueviolet', marker='D', linestyle='-.', label='without considering bzt')
# plt.legend()  # 显示图例
# plt.xlabel("percentage of vehicles being affected by byzantine attack ")  # X轴标签
# plt.ylabel("Test error rate")  # Y轴标签
# plt.show()

# y_14 = [0.8412, 0.9219, 0.9398, 0.9465, 0.9528, 0.9543, 0.9578, 0.9579, 0.9634, 0.9642]  # AFL_weight
# y_24 = [0.6971, 0.7932, 0.9008, 0.9286, 0.9268, 0.121, 0.0974, 0.0974, 0.0974, 0.6826] # AFL
# y_34 = [0.1090, 0.2327, 0.3794, 0.3445, 0.5717, 0.7295, 0.7535, 0.7728, 0.8059, 0.7764] # FL
# y_341 = [0.0871, 0.0778, 0.5219, 0.7320, 0.5941, 0.7547, 0.8398, 0.8626, 0.7905, 0.8508] # FL acc2
# y_3411 = [0.0772, 0.0861, 0.1659, 0.6048, 0.4298, 0.7412, 0.8047, 0.3231, 0.739, 0.7974]
# # plt.plot(x, y_14, color='orangered', marker='o', linestyle='-', label='本文方案')
# # plt.plot(x, y_24, color='blueviolet', marker='D', linestyle='-.', label='传统异步联邦方案')
# # plt.plot(x, y_3411, color='green', marker='*', linestyle=':', label='传统联邦方案')
#
# plt.plot(x, y_14, color='orangered', marker='o', linestyle='-', label='our scheme')
# plt.plot(x, y_24, color='blueviolet', marker='D', linestyle='-.', label='traditional AFL')
# plt.plot(x, y_3411, color='green', marker='*', linestyle=':', label='traditional FL')
#
# plt.legend()  # 显示图例
# # plt.xticks(x, names, rotation=45)
# # plt.xticks(x, names)
# plt.xticks(x, names)
# # plt.xlabel("步数")  # X轴标签
# # plt.ylabel("精度")  # Y轴标签
# plt.xlabel("number of step")  # X轴标签
# plt.ylabel("accuracy")  # Y轴标签
#
# plt.show()

# loss:
y_1 = [1.6568, 0.857782, 0.6168, 0.51871, 0.465, 0.415740, 0.4095, 0.3811, 0.347, 0.34820902]# AFL_weight
# y_2 = [1.7509, 1.0709, 1.2095, 0.7316, 0.6874, 0.8635, 2.1154, 2.3044, 2.3032, 2.1857]  # AFL
# y_3 = [2.1778, 2.2197, 1.9896, 1.8918, 1.8352, 1.6910, 1.6494, 1.6570, 1.8329, 1.8548]  # FL
# y_31 = [5.0825, 2.0824, 2.1407, 2.4323, 1.5807, 1.5878,1.5098, 1.4159, 1.3202, 1.3431]  # FL loss2
# y_311 = [2.2044, 2.3041, 2.3085, 2.8498, 3.0494, 1.8160, 1.8043, 3.3801, 1.6547, 1.4262]
#
y_4=[1.9061703976557811, 1.1038255333478226, 0.8496850455400397, 0.7563863726330642, 0.7327558676709315, 0.705055283297322, 0.6677244508598568, 0.6382514337847676, 0.6556666386184291, 0.6373043955089952]

y_5=[1.8487517900958639, 1.0703597134605431, 0.8535896766310026, 0.7766171828278117, 0.716093559456225, 0.6645781474792719, 0.6658517909914937, 0.6615171902207988, 0.6637005618377292, 0.6367255233603075]
# # plt.plot(x, y_1, color='orangered', marker='o', linestyle='-', label='本文方案')
# # plt.plot(x, y_2, color='blueviolet', marker='D', linestyle='-.', label='传统异步联邦方案')
# # plt.plot(x, y_311, color='green', marker='*', linestyle=':', label='传统联邦方案')
#
# plt.plot(x, y_1, color='orangered', marker='o', linestyle='-', label='our scheme')
# plt.plot(x, y_2, color='blueviolet', marker='D', linestyle='-.', label='traditional AFL')
# plt.plot(x, y_311, color='green', marker='*', linestyle=':', label='traditional FL')
#
# plt.legend()  # 显示图例
# # plt.xticks(x, names, rotation=45)
# # plt.xticks(x, names)
# plt.xticks(x, names)
# # plt.xlabel("步数")  # X轴标签
# # plt.ylabel("损失")  # Y轴标签
# plt.xlabel("number of step")  # X轴标签
# plt.ylabel("loss")  # Y轴标签
#
# plt.show()
# #
plt.plot(x, y_1, color='orangered', marker='o', linestyle='-', label='our scheme')
plt.plot(x, y_4, color='blueviolet', marker='D', linestyle=':', label='our scheme without lt')
plt.plot(x, y_5, color='green', marker='+', linestyle=':', label='our scheme without ct')
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
#[2.2044, 2.3041, 2.3085, 2.8498, 3.0494, 1.8160, 1.8043, 8.3801, 1.6547, 1.4262]
#[0.0772, 0.0861, 0.1659, 0.6048, 0.4298, 0.7412, 0.8047, 0.3231, 0.739, 0.7974]
