#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# sampling.py：IID/Non-IID 数据划分（复现论文核心数据场景）
# 修改说明：mnist_noniid 和 cifar_noniid 现已支持 classes_per_user 参数动态调整异构度

import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)  # 每个客户端的数据量
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        # 随机选择数据，确保每个客户端数据分布与全局一致
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users, classes_per_user=2):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset: MNIST 数据集
    :param num_users: 客户端数量
    :param classes_per_user: (新增) 每个客户端拥有的类别数，默认2
    :return:
    """
    # 动态计算总碎片数：用户数 * 每用户类别数
    # 例如：30用户 * 2类 = 60个碎片；30用户 * 1类 = 30个碎片
    num_shards = num_users * classes_per_user

    # 动态计算每个碎片的图片数量
    num_imgs = int(len(dataset) / num_shards)

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)

    # 兼容性处理：获取标签 (新版torchvision使用targets)
    if hasattr(dataset, 'targets'):
        labels = dataset.targets.numpy()
    else:
        labels = dataset.train_labels.numpy()

    # 关键：按标签排序（论文“ Non-IID”核心步骤）
    # 排序后数据结构类似：[0,0,0... 1,1,1... 9,9,9]
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 按标签升序排列
    idxs = idxs_labels[0, :]  # 排序后的样本索引

    # 分配逻辑
    for i in range(num_users):
        # 随机选择指定数量的碎片 (classes_per_user)
        # 由于数据已按标签排序，选不同的碎片大概率意味着选到了不同的类别
        rand_set = set(np.random.choice(idx_shard, classes_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)

        for rand in rand_set:
            # 拼接碎片样本索引，形成客户端数据
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    (保留此函数用于未来可能的对比实验：数据量偏斜场景)
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)

    if hasattr(dataset, 'targets'):
        labels = dataset.targets.numpy()
    else:
        labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    random_shard_size = np.random.randint(min_shard, max_shard + 1, size=num_users)
    random_shard_size = np.around(random_shard_size / sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    if sum(random_shard_size) > num_shards:
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

        random_shard_size = random_shard_size - 1

        for i in range(num_users):
            if len(idx_shard) == 0: continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard): shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    else:
        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

        if len(idx_shard) > 0:
            shard_size = len(idx_shard)
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users, classes_per_user=2):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param classes_per_user: (新增) 控制 CIFAR 的异构度
    """
    # 动态计算碎片，逻辑同 MNIST
    num_shards = num_users * classes_per_user
    num_imgs = int(len(dataset) / num_shards)

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)

    # 获取标签
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array(dataset.train_labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, classes_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


if __name__ == '__main__':
    # 简单的测试代码
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    # 测试默认参数 (2类)
    d = mnist_noniid(dataset_train, num, classes_per_user=2)
    print(f"Sampled user 0 data length: {len(d[0])}")