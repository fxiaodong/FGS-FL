#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# sampling.py：IID/Non-IID 数据划分（复现论文核心数据场景）
# 该文件实现了论文中 MNIST/CIFAR 的 IID/Non-IID 划分逻辑，是模拟 “去中心化数据分布” 的关键，核心函数对应论文实验设计：

import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)# 每个客户端的数据量（MNIST 60000/100=600）
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        # 随机选择数据，确保每个客户端数据分布与全局一致
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300     # 200个数据碎片，每个碎片300样本
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()    # 获取样本标签

    # 关键：按标签排序（论文“ Non-IID”核心步骤）
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()] # 按标签升序排列
    idxs = idxs_labels[0, :]      # 排序后的样本索引

    # 每个客户端分配2个碎片（论文设置），确保多数客户端仅含2类样本
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))   # 选2个碎片
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            # 拼接碎片样本索引，形成客户端数据
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    """
    生成非独立同分布（Non-IID）且客户端数据量不平衡的 MNIST 训练集划分，
    模拟真实联邦场景（如部分用户数据量大、部分用户数据量小，且各用户数据标签集中）。
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    # 总碎片数=1200，每个碎片含50张样本（1200*50=60000，覆盖全量训练集）
    num_shards, num_imgs = 1200, 50
    # 碎片索引列表（0~1199）
    idx_shard = [i for i in range(num_shards)]
    # 存储客户端-样本索引的字典（键=客户端ID，值=样本索引数组）
    dict_users = {i: np.array([]) for i in range(num_users)}
    # 全量样本的索引（0~59999）
    idxs = np.arange(num_shards*num_imgs)
    # 获取全量样本的标签（MNIST标签为0~9）
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    # 每个客户端最少/最多分配的碎片数

    min_shard = 1 # 确保每个客户端至少有1个碎片（避免无数据）
    max_shard = 30  # 最多分配30个碎片（制造数据量差异）

    #生成每个客户端的“目标碎片数”（确保总和=总碎片数1200）
    # 随机生成每个客户端的碎片数（范围1~30）
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    # 归一化：让碎片数总和=1200
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    # 转为整数（碎片数必须为整数）
    random_shard_size = random_shard_size.astype(int)

    # 碎片分配：分情况处理（避免碎片浪费）
    # 情况1：随机碎片总数 > 总碎片数（sum(random_shard_size) > 1200）
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # 第二步：分配剩余碎片
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    # 情况2：随机碎片总数 ≤ 总碎片数（sum(random_shard_size) ≤ 1200）
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users

# 生成独立同分布（IID） 的 CIFAR-10 训练集划分，
# 模拟 “理想联邦场景”—— 每个客户端的数据分布与全局一致，且样本量相等。
def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

# 生成非独立同分布（Non-IID）
# 的 CIFAR-10 训练集划分，
# 逻辑与mnist_noniid一致，但适配 CIFAR-10 的数据集规模（50000 张训练集）。
def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
