#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def init_gt_coords(args):
    """
    初始化GT坐标与发射功率
    区域: area_size x area_size (默认 200m * 200m)
    """
    np.random.seed(args.seed)
    # 生成 M 个终端的 (x, y) 坐标
    gt_coords = np.random.uniform(0, args.area_size, (args.num_users, 2))
    # 为每个 GT 分配随机发射功率 [P_min, P_max]
    gt_powers = np.random.uniform(args.gt_power_min, args.gt_power_max, args.num_users)
    return gt_coords, gt_powers


def calc_distance(uav_coord, gt_coords, args):
    """
    计算UAV（悬停在高度 H）与所有 GT 之间的欧几里得距离 (三维)
    """
    uav_x, uav_y = uav_coord
    # 利用 numpy 广播机制快速计算
    distances = np.sqrt(
        (gt_coords[:, 0] - uav_x) ** 2 +
        (gt_coords[:, 1] - uav_y) ** 2 +
        args.uav_height ** 2
    )
    return distances


def calc_path_loss(args, distance):
    """
    计算平均路径损耗：严格对应 csmcFL 论文公式 (12)-(14)
    适配实验 6：信道恶化实验 (force_nlos)
    """
    c = 3e8  # 光速
    fc = args.carrier_freq
    distance = np.maximum(distance, 1.0)  # 避免分母为0

    # --- 1. 计算 LoS 和 NLoS 的物理路径损耗 (Eq. 12 & 13) ---
    # PL = 20*log10(4*pi*fc*d/c) + zeta
    free_space_loss = 20 * np.log10(4 * np.pi * fc * distance / c)
    pl_los = free_space_loss + args.zeta_los
    pl_nlos = free_space_loss + args.zeta_nlos

    # --- 2. 计算视距概率 Pr_los (Eq. 14 核心描述) ---
    if args.force_nlos:
        # 【实验 6 逻辑】强制设定的 NLoS 概率，模拟环境恶化
        pr_nlos = args.nlos_prob
        pr_los = 1.0 - pr_nlos
    else:
        # 【标准物理逻辑】根据仰角计算视距概率
        # theta = arcsin(H/d)
        theta = np.arcsin(np.clip(args.uav_height / distance, -1, 1)) * 180 / np.pi
        # Pr_los = 1 / (1 + a * exp(-b * (theta - a)))
        pr_los = 1.0 / (1.0 + args.city_a * np.exp(-args.city_b * (theta - args.city_a)))
        pr_nlos = 1.0 - pr_los

    # --- 3. 平均路径损耗 (Eq. 14) ---
    pl_avg = pr_los * pl_los + pr_nlos * pl_nlos
    return pl_avg


def calc_gt_rate(args, gt_power, pl):
    """
    计算 GT 到 UAV 的上行传输速率 (Eq. 15)
    """
    # 将 dBm 转换为 mW
    gt_power_mw = 10 ** (gt_power / 10)
    noise_power_mw = 10 ** (args.noise_power / 10)

    # 计算接收功率 (mW) = 发射功率 / 路损(倍数)
    received_power = gt_power_mw / (10 ** (pl / 10))

    # 香农公式: R = B * log2(1 + SNR)
    rate = args.bandwidth * np.log2(1 + received_power / noise_power_mw)

    # 保证最小速率 R_min (适配代码逻辑鲁棒性)
    return np.maximum(rate, args.R_min)


def pso_uav_optimize(args, gt_coords, gt_powers):
    """
    PSO 粒子群算法：优化 UAV 的 (x, y) 坐标以最大化系统平均速率 (csmcFL 问题 P2)
    """
    np.random.seed(args.seed)

    # 初始化粒子位置与速度
    particles = np.random.uniform(0, args.area_size, (args.pso_particles, 2))
    velocities = np.random.uniform(-5, 5, (args.pso_particles, 2))

    pbest_pos = particles.copy()
    pbest_val = np.zeros(args.pso_particles)

    # 初始最优搜索
    for i in range(args.pso_particles):
        dists = calc_distance(particles[i], gt_coords, args)
        pl = calc_path_loss(args, dists)
        rates = calc_gt_rate(args, gt_powers, pl)
        pbest_val[i] = np.mean(rates)  # 目标：最大化平均速率

    gbest_pos = pbest_pos[np.argmax(pbest_val)].copy()
    gbest_val = np.max(pbest_val)

    # 迭代优化
    for iter_idx in range(args.pso_iter):
        for i in range(args.pso_particles):
            r1, r2 = np.random.rand(2)
            # 速度更新
            velocities[i] = args.pso_w * velocities[i] + \
                            args.pso_c1 * r1 * (pbest_pos[i] - particles[i]) + \
                            args.pso_c2 * r2 * (gbest_pos - particles[i])

            # 位置更新与边界处理
            particles[i] = np.clip(particles[i] + velocities[i], 0, args.area_size)

            # 评估新位置
            dists = calc_distance(particles[i], gt_coords, args)
            pl = calc_path_loss(args, dists)
            rates = calc_gt_rate(args, gt_powers, pl)
            current_fitness = np.mean(rates)

            if current_fitness > pbest_val[i]:
                pbest_val[i] = current_fitness
                pbest_pos[i] = particles[i].copy()

            if current_fitness > gbest_val:
                gbest_val = current_fitness
                gbest_pos = particles[i].copy()

    # 计算最终最优位置下的所有 GT 的具体速率
    final_dists = calc_distance(gbest_pos, gt_coords, args)
    final_pl = calc_path_loss(args, final_dists)
    final_rates = calc_gt_rate(args, gt_powers, final_pl)

    if getattr(args, 'verbose', 1):  # 如果没有 verbose 属性，默认视为 1 (开启)
        print(f"PSO 优化完成: UAV位置={gbest_pos}, 平均速率={gbest_val / 1e3:.2f} kbps")
        if getattr(args, 'force_nlos', False):
            print(f"警告: 当前处于实验6模式 (强制 NLoS 概率={args.nlos_prob})")

    return gbest_pos, final_rates