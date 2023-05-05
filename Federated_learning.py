import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from local_Update import LocalUpdate, test_inference, get_dataset, average_weights, exp_details
from local_model import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, cifar_iid, cifar_noniid

start_time = time.time()

# define paths
path_project = os.path.abspath('..')
logger = SummaryWriter('../logs')

args = args_parser()
exp_details(args)

if args.gpu == 1:
    torch.cuda.set_device(0)
device = 'cuda' if args.gpu else 'cpu'

# load dataset and user groups
trdata, tsdata, usgrp = get_dataset(args)     # get_dataset ：return train_dataset, test_dataset, user_groups

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
# print('初始化的全局模型glmodel为：', glmodel)

# copy weights
glweights = glmodel.state_dict()
# print('初始化的全局模型权重glweights为：', glweights)
# Training
trloss, tracc = [], []
vlacc, net_ = [], []
cvloss, cvacc = [], []
print_epoch = 2
vllossp, cnt = 0, 0

for ep in tqdm(range(args.epochs)):  # tqdm用于显示进度条部分

    locweights, locloss = [], []
    print(f'\n | Global Training Round : {ep+1} |\n')

    glmodel.train()
    m = max(int(args.frac * args.num_users), 1)
    user_id = np.random.choice(range(args.num_users), m, replace=False)
    print("user_id is", user_id)
    for j in user_id:
        locmdl = LocalUpdate(args=args, dataset=trdata, idxs=usgrp[j], logger=logger)
        w, loss, locmodel = locmdl.update_weights(model=copy.deepcopy(glmodel), global_round=ep, index=j)
        locweights.append(copy.deepcopy(w))
        locloss.append(copy.deepcopy(loss))

    # update global weights
    glweights = average_weights(locweights)      # 用的FL平均，将本地所有权重加权求平均得到全局权重
    # print('glweights为：', glweights)

    # update global weights
    glmodel.load_state_dict(glweights)   # load_state_dict用于将预训练的参数权重加载到新的模型之中

    avg_loss = sum(locloss) / len(locloss)
    trloss.append(avg_loss)     # 训练损失，为本地损失的加权平均
    # print('trloss为：', trloss)

    # Calculate avg training accuracy over all users at every epoch
    #  每个ep，使用所有用户来进行评估全局模型（相当于部分客户训练出结果，然后所有用户评估模型，最终得到一个训练精度）
    epacc, eploss = [], []
    glmodel.eval()
    for q in range(args.num_users):
        # locmdl = LocalUpdate(args=args, dataset=trdata, idxs=usgrp[j], logger=logger)
        locmdl = LocalUpdate(args=args, dataset=trdata, idxs=usgrp[q], logger=logger)
        acc, loss = locmdl.inference(model=glmodel)
        epacc.append(acc)
        eploss.append(loss)
    print('eploss为：', eploss)
    print('epacc为：', epacc)
    tracc.append(sum(epacc)/len(epacc))    # 计算平均每个ep的精度
    # print('tracc为：', tracc)

    # # print global training loss after every 'i' rounds
    # if (ep+1) % print_epoch == 0:          # print_epoch = 2
    #     print(f' \nAvg Training Stats after {ep+1} global rounds:')
    #     print(f'Training Loss : {np.mean(np.array(trloss))}')
    #     print('Train Accuracy: {:.2f}% \n'.format(100*tracc[-1]))

    # print global training loss after every 'i' rounds

    print(f' \nAvg Training Stats after {ep+1} global rounds:')
    print(f'Training Loss : {np.mean(np.array(trloss))}')
    print('Train Accuracy: {:.2f}% \n'.format(100*tracc[-1]))

print('trloss is :', trloss)
print('tracc is:', tracc)

# Test inference after completion of training
tsacc, tsloss = test_inference(args, glmodel, tsdata)       # 测试是使用测试集来进行测试的

print(f' \n Results after {args.epochs} global rounds of training:')
print("|---- Avg Train Accuracy: {:.2f}%".format(100*tracc[-1]))
print("|---- Test Accuracy: {:.2f}%".format(100*tsacc))

# Saving the objects train_loss and train_accuracy:
fname = 'results/models/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    format(args.dataset, args.model, args.epochs, args.frac, args.iid,
           args.local_ep, args.local_bs)

with open(fname, 'wb') as f:
    pickle.dump([trloss, tracc], f)

print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

# PLOTTING (optional)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# Plot Loss curve
plt.figure()
plt.title('Training Loss using Federated Learning')
plt.plot(range(len(trloss)), trloss, color='b')
plt.ylabel('Training loss')
plt.xlabel('Num of Rounds')
plt.savefig('results/FL_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
             format(args.dataset, args.model, args.epochs, args.frac,
                   args.iid, args.local_ep, args.local_bs))
#
# # Plot Average Accuracy vs Communication rounds
plt.figure()
plt.title('Average Accuracy using Federated Learning')
plt.plot(range(len(tracc)), tracc, color='g')
plt.ylabel('Average Accuracy')
plt.xlabel('Num of Rounds')
plt.savefig('results/FL_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
            format(args.dataset, args.model, args.epochs, args.frac,
                   args.iid, args.local_ep, args.local_bs))