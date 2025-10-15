#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# options.py：超参数配置（对应论文 FedAvg 关键参数）
# 该文件定义了所有实验参数，直接映射论文中的核心变量，是控制实验场景的入口

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # 1. 联邦核心参数（对应论文 FedAvg 算法参数）
    # 全局轮数
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    # 客户端数量
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # 2.模型与数据集参数（对应论文实验设置）
    # --model: 指定要使用的模型名称，默认为'mlp'
    # --kernel_num: 指定每种卷积核的数量，默认为9
    # --kernel_sizes: 指定用于卷积操作的卷积核尺寸，以逗号分隔，默认为'3,4,5'
    # --num_channels: 指定图像的通道数，默认为1
    # --norm: 指定使用的归一化方法，可选'batch_norm'、'layer_norm'或'None'，默认为'batch_norm'
    # --num_filters: 指定卷积网络中滤波器的数量，mini-imagenet建议32，omiglot建议64，默认为32
    # --max_pool: 指定是否使用最大池化而不是步幅卷积，默认为'True'
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    # 添加数据集名称参数，默认为'mnist'
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                           of dataset")
    # 添加类别数量参数，默认为10
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                           of classes")
    # 添加GPU设置参数，用于指定使用的GPU设备，默认使用CPU
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                           to a specific GPU ID. Default set to use CPU.")
    # 添加优化器类型参数，默认为'sgd'
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                           of optimizer")
    # 添加数据分布设置参数，1表示IID分布，0表示non-IID分布，默认为IID
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    # 添加non-IID设置下的数据分割方式参数，0表示等分，1表示不等分
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                           non-i.i.d setting (use 0 for equal splits)')
    # 添加早停轮数参数，用于控制训练提前停止的轮数，默认为10
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    # 添加详细输出级别参数，控制程序运行时的输出详细程度，默认为1
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    # 添加随机种子参数，用于保证实验可重现性，默认为1
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    # 解析命令行参数
    args = parser.parse_args()
    return args
