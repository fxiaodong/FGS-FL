#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, copy, time, datetime
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import gc
from options import args_parser
from update import LocalUpdate, test_inference
from utils import (get_dataset, init_uav_and_gt, choose_optimal_ro,
                   aggregate_payloads, exp_details, build_global_model)
from models import get_fc_layer


def compute_model_size_bits(model):
    return sum(p.numel() for p in model.parameters()) * 32


def main():
    args = args_parser()
    if args.lr < 0.01: args.lr = 0.01
    device = torch.device(f'cuda:{args.gpu}' if args.gpu and args.gpu != 'None' else 'cpu')

    # 1. Init Data & Env
    train_dataset, test_dataset, user_groups = get_dataset(args)
    args.uav_coord, args.gt_coords, args.gt_rates = init_uav_and_gt(args)

    # 2. 模型分析
    tmp_model = build_global_model(args).to(device)
    full_model_size_bits = compute_model_size_bits(tmp_model)
    target_layers = [lyr.strip() for lyr in args.compress_layers.split(',') if lyr.strip() and lyr.strip() != 'None']

    # 3. 压缩率与 PSO 优化
    args.compression_rate = 1.0  # 默认为 1.0
    args.use_csmc = 1 if args.strategy in ['csmcFL', 'fgs_csmcFL'] else 0
    if args.use_csmc:
        _, args.optimal_X, _ = choose_optimal_ro(args, args.gt_rates, full_model_size_bits)

    exp_details(args)
    global_model = build_global_model(args).to(device)
    user_histories = {i: [] for i in range(args.num_users)}
    test_acc_list, cumulative_bits_list = [], []
    total_bits, best_acc = 0.0, 0.0

    # 4. Training Loop
    for epoch in range(args.epochs):
        if epoch > 0 and epoch % 20 == 0:
            args.lr = max(args.lr * 0.7, 0.004)
            print(f'>>> Learning rate adjusted to: {args.lr:.6f}')

        print(f'Round {epoch + 1}/{args.epochs} | Strategy: {args.strategy}')
        selected_users = np.arange(args.num_users)
        local_payloads, round_train_losses = [], []

        for idx in selected_users:
            # 预热判定
            is_comp = False if epoch < 40 else (args.use_csmc and args.optimal_X[idx] == 0)

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
            total_bits += full_model_size_bits * (args.compression_rate if is_comp else 1.0)

            # 每走完一个用户，显式清理
            del local_trainer
            gc.collect()

        # 5. 聚合与测试
        if local_payloads:
            agg_result = aggregate_payloads(local_payloads)
            global_model.load_state_dict(agg_result, strict=True)
            del local_payloads, agg_result

        acc, test_loss = test_inference(args, global_model, test_dataset)
        test_acc_list.append(acc)
        cumulative_bits_list.append(total_bits)
        if acc > best_acc: best_acc = acc

        print(f'   [Stat] Test Acc: {acc:.2%} | Best: {best_acc:.2%} | Comm: {total_bits / 1e6:.2f}Mb')

        # 每轮结束强制清理显存碎片
        torch.cuda.empty_cache()

    # 保存结果
    save_dir = '../save/results/'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    file_name = f"{args.strategy}_{args.dataset}_{datetime.datetime.now().strftime('%m%d_%H%M')}.pth"
    torch.save({'args': vars(args), 'test_acc': test_acc_list, 'cumulative_bits': cumulative_bits_list},
               os.path.join(save_dir, file_name))


if __name__ == '__main__':
    main()