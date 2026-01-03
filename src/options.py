#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # ========================== 1. 联邦学习核心参数 ==========================
    parser.add_argument('--epochs', type=int, default=30, help="全局聚合轮数（论文R=30）")
    parser.add_argument('--num_users', type=int, default=30, help="GT数量（论文M=30）")
    parser.add_argument('--frac', type=float, default=1.0, help="客户端选择比例（psFL使用，csmcFL由算法自动决定）")
    parser.add_argument('--local_ep', type=int, default=5, help="本地训练轮次（论文E=5）")
    parser.add_argument('--local_bs', type=int, default=16, help="本地批大小（论文B=16）")
    parser.add_argument('--lr', type=float, default=0.001, help="学习率（论文ε=0.001）")
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD动量')

    # ========================== 2. 模型与数据集参数 ==========================
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'mlp', 'lenet', 'alexnet'],
                        help='模型类型。cnn会根据数据集自动匹配论文结构的LeNet5或AlexNet')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fmnist', 'cifar'], help="数据集")
    parser.add_argument('--num_classes', type=int, default=10, help="类别数")
    parser.add_argument('--gpu', default='0', help="GPU ID (例如 '0')，若使用CPU请传 None")
    parser.add_argument('--iid', type=int, default=1, help='1=IID, 0=Non-IID')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    parser.add_argument('--classes_per_user', type=int, default=2,
                        help='Non-IID 模式下每个客户端拥有的类别数 (1-10)')

    # ========================== 3. csmcFL 联合优化与压缩 ==========================
    parser.add_argument('--strategy', type=str, default='fgs_csmcFL',
                        choices=['tFL', 'psFL', 'csmcFL', 'dp_fedavg', 'dp_fedavg_comp', 'fgs_csmcFL', 'fgs_fl'],
                        help='实验策略选择: fgs_csmcFL(Fed-LRP), fgs_fl(原论文无压缩对比)')
    # 修改 compress_layers 描述，明确 None 的用法
    parser.add_argument('--compress_layers', type=str, default='fc1',
                        help='SVD压缩的目标层。设为 None 则执行原论文 fgs_fl 全量加噪模式')
    parser.add_argument('--compress_rank', type=int, default=10, help='SVD初始秩λ')
    parser.add_argument('--compress_rank_max', type=int, default=50, help='SVD最大允许秩上限')
    parser.add_argument('--R_min', type=float, default=5e4, help='csmcFL要求的最小传输速率（bit/s）')

    # ========================== 4. UAV 与 无线信道参数 (论文 Eq.12-15) ==========================

    parser.add_argument('--area_size', type=float, default=200.0, help='GT分布区域 (m)')
    parser.add_argument('--uav_height', type=float, default=100.0, help='UAV悬停高度 (m)')
    parser.add_argument('--carrier_freq', type=float, default=2.0e9, help='载波频率 (fc=2GHz)')
    parser.add_argument('--bandwidth', type=float, default=2.0e6, help='带宽 (B=2MHz)')
    parser.add_argument('--noise_power', type=float, default=-96.0, help='噪声功率 (N=-96dBm)')
    parser.add_argument('--gt_power_min', type=float, default=0.0, help='GT最小发射功率 (0dBm)')
    parser.add_argument('--gt_power_max', type=float, default=10.0, help='GT最大发射功率 (10dBm)')

    # 路径损耗环境参数 (Eq.14)
    parser.add_argument('--city_a', type=float, default=9.61, help='环境参数 a')
    parser.add_argument('--city_b', type=float, default=0.16, help='环境参数 b')
    parser.add_argument('--zeta_los', type=float, default=1.0, help='LoS 附加损耗 (dB)')
    parser.add_argument('--zeta_nlos', type=float, default=20.0, help='NLoS 附加损耗 (dB)')

    # 实验 6：信道恶化人工干预
    parser.add_argument('--force_nlos', action='store_true', help='实验6开关：强制固定NLoS概率')
    parser.add_argument('--nlos_prob', type=float, default=0.2, help='实验6：手动设定的NLoS概率值 (0.2, 0.5, 0.8)')

    # ========================== 5. FGS 与 差异隐私参数 (ESWA 2025) ==========================
    parser.add_argument('--epsilon', type=float, default=4.0, help='隐私预算 ε')
    parser.add_argument('--dp_type', type=str, default='gsr',
                        choices=['grad_clip', 'post_train', 'gsr'],
                        help='实验8：DP加噪点 (GSR为FGS-FL核心)')
    parser.add_argument('--rho', type=float, default=0.05, help='FGO平坦化扰动半径')

    # ========================== 6. 算法逻辑开关 (用于消融实验 7) ==========================
    parser.add_argument('--ablation', type=str, default='all',
                        choices=['none', 'client_sel', 'compression', 'all'],
                        help='实验7：none(Base A), client_sel(Base B), compression(Base C), all(完整csmcFL)')
    parser.add_argument('--target_acc', type=float, default=0.85, help='实验3：评估收敛速度的目标精度')
    parser.add_argument('--conv_threshold', type=float, default=1e-4, help='收敛判定的损失变化阈值（默认1e-4）')
    parser.add_argument('--patience', type=int, default=3, help='连续满足收敛条件的轮数（默认3轮）')
    # ========================== 7. PSO 优化器参数 ==========================
    parser.add_argument('--pso_iter', type=int, default=50, help='PSO迭代次数')
    parser.add_argument('--pso_particles', type=int, default=30, help='PSO粒子数')
    parser.add_argument('--pso_w', type=float, default=0.7, help='惯性权重')
    parser.add_argument('--pso_c1', type=float, default=2.0, help='个体因子')
    parser.add_argument('--pso_c2', type=float, default=2.0, help='全局因子')
    # ========================== 8. 日志与调试参数 ==========================
    parser.add_argument('--verbose', type=int, default=1,
                        help='是否打印详细日志 (1=开启, 0=关闭)')
    parser.add_argument('--log_dir', type=str, default='../logs', help='Tensorboard 日志目录')

    return parser.parse_args()

    return parser.parse_args()