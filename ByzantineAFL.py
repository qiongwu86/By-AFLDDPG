# FL相关

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # -1表示不使用GPU   0/1为显卡名称（使用哪个显卡） 后面联邦学习里面设置了
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from scipy import special as sp
from scipy.constants import pi

import torch
from tensorboardX import SummaryWriter

from local_Update import LocalUpdate, test_inference, get_dataset, average_weights, exp_details, asy_average_weights, asy_average_weights_weight
from local_model import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, cifar_iid, cifar_noniid

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# DDPG相关

from ddpg_env_10_4_new import *
from parameters import *
from agent import *
import tensorflow as tf

import tflearn
import ipdb as pdb
from options import *

# AFL相关
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 防止画图的时候图形界面不显示（在pycharm上运行时需要加上这行代码）

start_time = time.time()

# define paths
path_project = os.path.abspath('..')
logger = SummaryWriter('../logs')

args = args_parser()
exp_details(args)

if args.gpu == 1:
    torch.cuda.set_device(0)
device = 'cuda' if args.gpu else 'cpu'


MAX_EPISODE = 1
MAX_EPISODE_LEN = 10
tracc111 = []
total_time = []
# byzantine指标参数
lampda = 2




# 开始训练episode
for ep in tqdm(range(MAX_EPISODE)):  # (多少个episode)
    # 每个episode开始AFL进行模型初始化
    # load dataset and user groups
    trdata, tsdata, usgrp = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            glmodel = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            glmodel = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            glmodel = CNNCifar(args=args)
    elif args.model == 'mlp':
        imsize = trdata[0][0].shape
        input_len = 1
        for x in imsize:
            input_len *= x
            glmodel = MLP(dim_in=input_len, dim_hidden=64,
                          dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    glmodel.to(device)
    glmodel.train()

    vehicle_model = []
    for i in range(args.num_users):
        vehicle_model.append(copy.deepcopy(glmodel))   # 初始化的全局模型下发到车

    # copy weights
    glweights = glmodel.state_dict()

    # Training
    trloss, tracc = [], []
    tr_step_loss = []
    tr_step_acc = []
    vlacc, net_ = [], []
    cvloss, cvacc = [], []
    print_epoch = 2
    vllossp, cnt = 0, 0

    glmodel.train()
    user_id = range(args.num_users-1)


    # 开始step
    tsacc1 = []
    AFL_loss1 = []
    cost1 = []

    #######################
    # 计算本地更新权重比例：

    # 每辆车可用计算资源delta
    mu1, sigma1 = 1.5e+9, 1e+8
    lower, upper = mu1 - 2 * sigma1, mu1 + 2 * sigma1  # 截断在[μ-2σ, μ+2σ]
    x = stats.truncnorm((lower - mu1) / sigma1, (upper - mu1) / sigma1, loc=mu1, scale=sigma1)
    delta = x.rvs(args.num_users)  # 总共得到每辆车的计算资源，车辆数目为clients_num来控制

    # 本地计算时间比例

    CPUcycles = 1e+6  # 10^6
    kexi = 0.9
    localtime = [0] * args.num_users
    beta_lt = [0] * args.num_users
    for i in range(args.num_users):
        localtime[i] = (1000 + 300 * i) * CPUcycles / delta[i] - 0.5
        beta_lt[i] = kexi ** localtime[i]


    # 进行AR信道建模
    def complexGaussian1(row=1, col=1, amp=1.0):
        real = np.random.normal(size=[row, col])[0] * np.sqrt(0.5)
        # np.random.normal(size=[row,col])生成数据2维，第一维度包含row个数据，每个数据中又包含col个数据
        # np.sqrt(A)求A的开方
        img = np.random.normal(size=[row, col])[0] * np.sqrt(0.5)
        return amp * (real + 1j * img)  # amp相当于系数，后面计算时替换成了根号下1-rou平方    (real + 1j*img)即为误差向量e(t)


    aaa = complexGaussian1(1, 1)[0]
    H1 = [aaa] * args.num_users

    # 车辆x轴坐标位置
    alpha = 2
    path_loss = [0] * args.num_users
    dis = [0] * args.num_users
    for i in range(args.num_users):  # i从 0 - (args.num_users-1)
        dis[i] = -500 + 50 * i  # 车辆初始位置
        path_loss[i] = 1 / np.power(np.linalg.norm(dis[i]), alpha)

    # 计算ρi
    Hight_RSU = 10
    width_lane = 5
    velocity = 20
    lamda = 7
    x_0 = np.array([1, 0, 0])
    P_B = np.array([0, 0, Hight_RSU])
    P_m = [0] * args.num_users
    rho = [0] * args.num_users
    for i in range(args.num_users):
        P_m[i] = np.array([dis[i], width_lane, Hight_RSU])
        rho[i] = sp.j0(2 * pi * velocity * np.dot(x_0, (P_B - P_m[i])) / (np.linalg.norm(P_B - P_m[i]) * lamda))

    # 计算H信道增益
    for i in range(args.num_users):
        # H1[i] = rho[i] * H1[i] + complexGaussian(1, 1, np.sqrt(1 - rho[i] * rho[i]))
        H1[i] = rho[i] * H1[i] + complexGaussian(1, 1, np.sqrt(1 - rho[i] * rho[i]))[0]
        ddd = abs(H1[i])

    # 计算传输速率
    transpower = 250
    sigma2 = 1e-9
    bandwidth = 1e+3  # HZ
    tr1 = [0] * args.num_users
    sinr1 = [0] * args.num_users
    for i in range(args.num_users):
        sinr1[i] = transpower * abs(H1[i]) * path_loss[i] / sigma2  # abs()即可求绝对值，也可以求复数的模 #因为神经网络里输入的值不能为复数
        tr1[i] = np.log2(1 + sinr1[i]) * bandwidth

    # 时隙t车辆i的通信时间,w_size为t时隙所学习的本地模型参数的大小，tr传输速率
    w_size = 5000  # 本地模型参数大小（5kbits）(香农公式传输速率单位为bit/s)
    c_c = [0] * args.num_users
    for i in range(args.num_users):
        c_c[i] = w_size / tr1[i] - 0.5

    beta_ct = [0] * args.num_users
    epuxilong = 0.9
    for i in range(args.num_users):
        beta_ct[i] = epuxilong ** c_c[i]

    #########################


    for j in range(MAX_EPISODE_LEN):
        ep_start_time = time.time()

        # 每个step开始，RSU处的计算的参考模型更新一次
        server_model = LocalUpdate(args=args, dataset=trdata, idxs=usgrp[5], logger=logger)  # 先暂时将测试样本数定为用第五个用户的数据量大小
        s_w, s_loss, s_model = server_model.asyupdate_weights(model=copy.deepcopy(glmodel), global_round=ep, index=j)
        print('server loss is:', s_loss)



        print("the j-th step, j=", j)

        # 开始AFL训练
        # 全部车进行本地loss计算
        locloss = []
        for aa in user_id:
            if aa != 3:
                local_net = copy.deepcopy(vehicle_model[aa])
                local_net.to(device)
                locmdl = LocalUpdate(args=args, dataset=trdata, idxs=usgrp[aa], logger=logger)
                w, loss, localmodel = locmdl.asyupdate_weights(model=copy.deepcopy(glmodel), global_round=ep, index=aa)

                print("the number of vehicle aa is:", aa)
                print("local loss:", loss)
                # 进行byzantine筛选
                # 对向量求模值：y = [4, 3]   np.linalg.norm(y)
                 # if np.linalg.norm(loss - s_loss) <= lampda * s_loss:
                if loss <= lampda * s_loss:
                    locloss.append(loss)
                    glmodel, glweights = asy_average_weights_weight(vehicle_idx=aa, global_model=glmodel, local_model=localmodel,
                                                         gamma=args.gamma,local_param2=beta_lt[aa], local_param3=beta_ct[aa])
                    print("the number of vehicle that update the global model is:", aa)

        # print("locloss is :", locloss)
        avg_loss = sum(locloss) / len(locloss)
        # print("avg_loss is :", avg_loss)
        AFL_loss1.append(avg_loss)
        # 全部训练完后把最终训练好的全局模型发给所有车辆（下一个step同样起跑线）
        for iz in range(args.num_users):
            vehicle_model[iz] = copy.deepcopy(glmodel)



        # Calculate avg training accuracy over all users at every epoch
        epacc11, eploss11 = [], []
        glmodel.eval()
        for q in range(args.num_users):
            locmdl = LocalUpdate(args=args, dataset=trdata,
                                 idxs=usgrp[q], logger=logger)
            acc, loss = locmdl.inference(model=glmodel)
            epacc11.append(acc)
            eploss11.append(loss)
        tracc111.append(sum(epacc11) / len(epacc11))


        # 每个step后计算测试精度和loss(loss为一个step里AFL完的全局模型的loss)
        tsacc, tsloss = test_inference(args, glmodel, tsdata)
        tsacc1.append(tsacc)


        ep_time = time.time() - ep_start_time
        total_time.append(ep_time)


    print("step test acc 即 tsacc1 is ", tsacc1)
    print('AFL_loss1 is :', AFL_loss1)


print('total_time is:', total_time)
# Test inference after completion of training
tsacc_final, tsloss_final = test_inference(args, glmodel, tsdata)

print(f' \n Results after {MAX_EPISODE} epoch rounds of training:')
# print("|---- Avg Train Accuracy: {:.2f}%".format(100*tracc[-1]))
print("|---- Test Accuracy: {:.2f}%".format(100 * tsacc_final))
print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))