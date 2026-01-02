import copy
import torch
import numpy as np
from torch import nn
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, cifar_iid, cifar_noniid
from models import (replace_fc_with_lowrank, get_fc_rank, get_fc_layer,
                    CNNCifar, CNNMnist, CNNFashion_Mnist, MLP, LeNet5, AlexNet)
from uav_utils import init_gt_coords, pso_uav_optimize


def get_dataset(args):
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)
        args.num_channels = 3
        if args.iid:
            raw_groups = cifar_iid(train_dataset, args.num_users)
        else:
            raw_groups = cifar_noniid(train_dataset, args.num_users, classes_per_user=args.classes_per_user)
    else:
        data_dir = f'../data/{args.dataset}/'
        dataset_cls = datasets.MNIST if args.dataset == 'mnist' else datasets.FashionMNIST
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = dataset_cls(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = dataset_cls(data_dir, train=False, download=True, transform=apply_transform)
        args.num_channels = 1
        if args.iid:
            raw_groups = mnist_iid(train_dataset, args.num_users)
        else:
            raw_groups = mnist_noniid(train_dataset, args.num_users, classes_per_user=args.classes_per_user)

    user_groups = {int(k): list(v) for k, v in raw_groups.items()}
    return train_dataset, test_dataset, user_groups


def init_uav_and_gt(args):
    gt_coords, gt_powers = init_gt_coords(args)
    uav_coord, gt_rates = pso_uav_optimize(args, gt_coords, gt_powers)
    gt_rates = np.maximum(gt_rates, 1.0)
    return uav_coord, gt_coords, gt_rates


def aggregate_payloads(payloads):
    if not payloads: return {}
    total_samples = sum(int(p.get('n_samples', 1)) for p in payloads)
    agg_state = {}
    for p in payloads:
        state = p['state']
        weight = float(p.get('n_samples', 1)) / float(total_samples)
        user_restored_state = {}
        compressed_keys = set()

        # 1. 解压：处理 SVD 分解的权重
        for k in state.keys():
            if '.W1' in k:
                base_name = k.replace('.W1', '')
                if base_name + '.W2' in state:
                    w1 = state[base_name + '.W1']
                    w2 = state[base_name + '.W2']
                    user_restored_state[base_name + '.weight'] = torch.matmul(w1, w2)
                    compressed_keys.add(base_name + '.W1')
                    compressed_keys.add(base_name + '.W2')

        # 2. 合并：加入 bias、BN 参数和未压缩的层
        for k, v in state.items():
            if k not in compressed_keys:
                user_restored_state[k] = v

        # 3. 聚合所有参数（包括 BN 的 running_mean 等）
        for k, v in user_restored_state.items():
            if k not in agg_state:
                agg_state[k] = v.float() * weight
            else:
                agg_state[k] += v.float() * weight
    return agg_state


def build_global_model(args):
    if args.model == 'cnn':
        if args.dataset == 'cifar':
            model = AlexNet(args)
        elif args.dataset in ['mnist', 'fmnist']:
            model = LeNet5(args)
        else:
            model = CNNCifar(args)
    elif args.model == 'lenet':
        model = LeNet5(args)
    elif args.model == 'alexnet':
        model = AlexNet(args)
    elif args.model == 'mlp':
        dim_in = 3072 if args.dataset == 'cifar' else 784
        model = MLP(dim_in=dim_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    return model.to(torch.device('cuda' if args.gpu and args.gpu != 'None' else 'cpu'))


def choose_optimal_ro(args, gt_rates, model_size_bits):
    candidate_ro = np.unique(gt_rates[gt_rates > 0])
    if candidate_ro.size == 0: return np.mean(gt_rates), np.ones(args.num_users), 0
    min_time, optimal_ro, optimal_X = float('inf'), float(np.mean(gt_rates)), np.ones(args.num_users)
    for ro in candidate_ro:
        X = (gt_rates >= ro).astype(int)
        if X.sum() == 0: continue
        T = calculate_fl_time(args, X, gt_rates, model_size_bits)
        if T < min_time:
            min_time, optimal_ro, optimal_X = T, float(ro), X.copy()
    return optimal_ro, optimal_X, min_time


def calculate_fl_time(args, X, gt_rates, model_size_bits):
    model = build_global_model(args)
    # fc_layer = get_fc_layer(model, args.compress_layers)
    # 修改为只取第一层名，避免解析失败
    first_layer = args.compress_layers.split(',')[0].strip()
    fc_layer = get_fc_layer(model, first_layer)
    lambda_total = max(1, get_fc_rank(fc_layer))
    eta = float(args.compress_rank) / lambda_total
    T_list = []
    for m in range(args.num_users):
        rate_m = max(float(gt_rates[m]), 1.0)
        upload_size = model_size_bits * X[m] + (model_size_bits * eta) * (1 - X[m])
        T_list.append(6.0 + upload_size / rate_m)
    return float(np.mean(T_list))


def exp_details(args):
    print(f"\n策略: {args.strategy} | 模型: {args.model} | 数据集: {args.dataset}")
    print(f"IID: {'是' if args.iid else '否'} | 每个用户类别数: {args.classes_per_user if not args.iid else 'N/A'}")
    print(f"压缩层: {args.compress_layers} | 目标秩: {args.compress_rank}")
    print(f"隐私预算 ε: {args.epsilon} | 噪声类型: {args.dp_type}\n")