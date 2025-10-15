#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# 数据集加载 + 权重聚合

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid

# 加载数据集并划分客户端数据
def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        # CIFAR-10 数据加载与预处理
        data_dir = '../data/cifar/'
        # 定义数据预处理变换：转换为张量并进行标准化
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # 创建CIFAR-10训练数据集
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        # 创建CIFAR-10测试数据集
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)

        # 根据配置将训练数据分发给不同用户
        if args.iid:
            # IID模式：独立同分布地将用户数据从CIFAR-10数据集中采样
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Non-IID模式：非独立同分布地将用户数据从CIFAR-10数据集中采样
            if args.unequal:
                # 为每个用户选择不等比例的数据分割
                raise NotImplementedError()
            else:
                # 为每个用户选择相等比例的数据分割
                user_groups = cifar_noniid(train_dataset, args.num_users)
    # MNIST/FMNIST 数据加载（28×28 灰度图）
    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups

# 聚合客户端权重（对应论文 “服务器加权平均”）
'''论文中是 “按客户端数据量 \(n_k/n\) 加权平均”，
此处简化为 “按客户端数量平均”，但核心逻辑一致（均为聚合客户端模型参数）。'''
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0]) # 初始化平均权重
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]# 累加所有客户端权重
        w_avg[key] = torch.div(w_avg[key], len(w)) # 按客户端数量平均
    return w_avg

# 打印实验的详细配置信息
def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
