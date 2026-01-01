#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, copy, time, datetime
import numpy as np
import torch
import gc
from options import args_parser
from update import LocalUpdate, test_inference
from utils import (get_dataset, init_uav_and_gt, choose_optimal_ro,
                   aggregate_payloads, exp_details, build_global_model)
from models import get_fc_rank, get_fc_layer


def compute_model_size_bits(model):
    return sum(p.numel() for p in model.parameters()) * 32


def main():
    args = args_parser()
    # 强制修正学习率以保证实验稳健性
    if args.lr < 0.01: args.lr = 0.01

    device = torch.device(f'cuda:{args.gpu}' if args.gpu and args.gpu != 'None' else 'cpu')
    gc.collect()
    torch.cuda.empty_cache()

    # ========== 1. Init Data & Env ==========
    train_dataset, test_dataset, user_groups = get_dataset(args)
    args.uav_coord, args.gt_coords, args.gt_rates = init_uav_and_gt(args)

    # ========== 2. 模型分析 ==========
    tmp_model = build_global_model(args).to(device)
    full_model_size_bits = compute_model_size_bits(tmp_model)

    # 【关键修改】：更鲁棒的层解析逻辑
    # 判定是否为“消融实验 3”：即关闭 LRP 压缩
    is_lrp_disabled = args.compress_layers is None or args.compress_layers.lower() in ['none', '']

    if is_lrp_disabled:
        target_layers = []
        args.compression_rate = 1.0
        # 如果不压缩，所有用户默认执行全量传输 (X=1)
        args.optimal_X = np.ones(args.num_users)
        print(">>> 模式：全量通信（Exp 3 消融实验：关闭 LRP 压缩）")
    else:
        target_layers = [lyr.strip() for lyr in args.compress_layers.split(',') if lyr.strip()]

    raw_total_fc_params = 0
    comp_total_fc_params = 0
    found_any_layer = False

    # ========== 3. 压缩率计算与 PSO 优化 ==========
    if target_layers:
        for layer_name in target_layers:
            fc_layer = get_fc_layer(tmp_model, layer_name)
            if fc_layer:
                in_feats, out_feats = fc_layer.in_features, fc_layer.out_features
                raw_total_fc_params += (in_feats * out_feats)
                actual_k = min(args.compress_rank, in_feats, out_feats)
                comp_total_fc_params += (in_feats * actual_k + actual_k * out_feats)
                found_any_layer = True

        if found_any_layer:
            fixed_part_bits = full_model_size_bits - (raw_total_fc_params * 32)
            compressed_full_size_bits = fixed_part_bits + (comp_total_fc_params * 32)
            args.compression_rate = compressed_full_size_bits / full_model_size_bits
            print(f">>> 模式：模型压缩开启，目标层: {target_layers}, 压缩率: {args.compression_rate:.4f}")

            # 只有在真正开启压缩层时，才运行 PSO 优化
            args.use_csmc = 1 if args.strategy in ['csmcFL', 'fgs_csmcFL'] else 0
            if args.use_csmc:
                _, args.optimal_X, _ = choose_optimal_ro(args, args.gt_rates, full_model_size_bits)
        else:
            args.compression_rate = 1.0
            args.optimal_X = np.ones(args.num_users)
            print(">>> 警告：未找到指定压缩层，切换为全量通信模式")

    # 打印实验细节
    exp_details(args)

    # 正式训练使用的全局模型
    global_model = build_global_model(args).to(device)
    user_histories = {i: [] for i in range(args.num_users)}

    test_acc_list, cumulative_bits_list = [], []
    total_bits = 0.0
    best_acc = 0.0

    # ========== 4. Training Loop ==========
    for epoch in range(args.epochs):
        # 学习率衰减策略
        if epoch > 0 and epoch % 20 == 0:
            new_lr = args.lr * 0.7
            args.lr = max(new_lr, 0.004)
            print(f'>>> Learning rate adjusted to: {args.lr:.6f}')

        print(f'Round {epoch + 1}/{args.epochs} | Strategy: {args.strategy}')

        # 动态调整压缩秩 (40-50轮预热后强化特征捕获)
        original_rank = args.compress_rank
        if 40 <= epoch < 50:
            args.compress_rank = original_rank * 2
        else:
            args.compress_rank = original_rank

        selected_users = np.arange(args.num_users)
        local_payloads, round_train_losses = [], []
        comp_user_count = 0

        for idx in selected_users:
            # 判定当前用户是否执行压缩
            # 逻辑：非预热期 & 策略开启压缩 & PSO 判定该用户信道差 (X=0)
            is_comp = False if epoch < 40 else (
                        not is_lrp_disabled and args.strategy in ['csmcFL', 'fgs_csmcFL'] and args.optimal_X[idx] == 0)
            if is_comp: comp_user_count += 1

            local_trainer = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            payload, loss, updated_history = local_trainer.update_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                compress=is_comp,
                epsilon=args.epsilon,
                history=user_histories[idx]
            )
            user_histories[idx] = updated_history
            local_payloads.append(payload)
            round_train_losses.append(loss)

            # 比特统计：基于实际是否压缩计算
            current_bits = full_model_size_bits * (args.compression_rate if is_comp else 1.0)
            total_bits += current_bits

        # 聚合
        if local_payloads:
            agg_result = aggregate_payloads(local_payloads)
            global_model.load_state_dict(agg_result, strict=True)

        # 测试
        acc, test_loss = test_inference(args, global_model, test_dataset)
        test_acc_list.append(acc)
        cumulative_bits_list.append(total_bits)

        if acc > best_acc:
            best_acc = acc

        avg_train_loss = sum(round_train_losses) / len(round_train_losses)
        print(f'   [Loss] Train: {avg_train_loss:.4f} | Test: {test_loss:.4f}')
        print(f'   [Stat] Test Acc: {acc:.2%} | Best: {best_acc:.2%} | Comm: {total_bits / 1e6:.2f}Mb' +
              (f" | CompUsers: {comp_user_count}" if epoch >= 40 else ""))

        # 还原 Rank
        args.compress_rank = original_rank

    # ========== 5. Save Results ==========
    save_dir = '../save/results/'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    iid_tag = 'iid' if args.iid else 'noniid'
    # 文件名增加 rho 标识，方便绘图脚本自动识别消融实验
    file_name = f"{args.strategy}_{args.dataset}_{iid_tag}_eps{args.epsilon}_rho{args.rho}_rank{args.compress_rank}_{timestamp}.pth"

    results = {'args': vars(args), 'test_acc': test_acc_list, 'cumulative_bits': cumulative_bits_list}
    torch.save(results, os.path.join(save_dir, file_name))
    print(f"\nTraining Finished. Max Acc: {max(test_acc_list):.2%}")


if __name__ == '__main__':
    main()