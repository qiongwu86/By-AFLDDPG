
# AFL_weight vs AFL : 本地cost对比
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

names = ['1', '2', '3', '4', '5', '6', '7','8','9','10']
x = range(len(names))
# x = range(,10)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字

# [1.9724, 2.3327, 3.0568, 4.5009, 2.0674, 2.6758,3.3001, 4.4889, 2.1113, 2.7187, 3.5455, 4.6525]

y_14 = [1.9724, 2.3327, 3.0568, 4.5009, 2.0674, 2.6758,3.3001, 4.4889, 2.1113, 2.7187]
y_24 = [13.6098, 15.9804, 16.0965, 19.0029, 18.1545, 16.0247, 16.2616, 15.8454, 15.1508, 15.5847]
y_25 = [15.171911001205444, 16.885496377944946, 16.130796432495117, 16.820281982421875, 16.877536058425903, 15.073806047439575, 15.213825464248657, 15.477785110473633, 17.414581060409546, 17.052588939666748]

# plt.plot(x, y_14, color='orangered', marker='o', linestyle='-', label='本文方案')
# plt.plot(x, y_25, color='blueviolet', marker='D', linestyle='-.', label='传统联邦方案')

plt.plot(x, y_14, color='orangered', marker='o', linestyle='-', label='our scheme')
plt.plot(x, y_25, color='blueviolet', marker='D', linestyle='-.', label='FL')

# plt.plot(x, y_3, color='green', marker='*', linestyle=':', label='C')
plt.legend()  # 显示图例
# plt.xticks(x, names, rotation=45)
# plt.xticks(x, names)
plt.xticks(x, names)
# plt.xlabel("全局模型更新次数")  # X轴标签
# plt.ylabel("执行时间")  # Y轴标签

plt.xlabel("number of global round")  # X轴标签
plt.ylabel("running time")  # Y轴标签

plt.show()


# our scheme:
# [14.674711227416992, 16.34528636932373, 15.749878644943237, 15.903467893600464, 15.636183261871338, 15.662116289138794, 15.79475712776184, 16.0849826335907, 16.29242968559265, 16.389168977737427]
[14.0524, 15.277143239974976, 15.223286390304565, 15.303074359893799, 15.2362]
[1.6704158782958984, 1.8199970722198486, 1.9699981212615967, 2.2499990463256836, 2.4775]