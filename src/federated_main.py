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

    # ========== 新增：停止条件参数（和之前版本对齐） ==========
    # 从命令行参数读取，无则用默认值
    target_acc = args.target_acc if hasattr(args, 'target_acc') else 0.85  # 目标准确率
    convergence_threshold = args.conv_threshold if hasattr(args, 'conv_threshold') else 1e-4  # 损失变化阈值
    patience = args.patience if hasattr(args, 'patience') else 3  # 连续满足收敛的轮数
    recent_test_losses = []  # 存储最近patience轮的测试损失
    stop_training = False  # 提前停止标志

    # 4. Training Loop
    for epoch in range(args.epochs):
        # 新增：满足停止条件则提前退出
        if stop_training:
            break

        # 学习率衰减策略（和之前版本一致）
        if epoch > 0 and epoch % 20 == 0:
            args.lr = max(args.lr * 0.7, 0.004)
            print(f'>>> Learning rate adjusted to: {args.lr:.6f}')

        print(f'Round {epoch + 1}/{args.epochs} | Strategy: {args.strategy}')
        selected_users = np.arange(args.num_users)
        local_payloads, round_train_losses = [], []
        comp_user_count = 0  # 新增：统计本轮压缩的用户数

        for idx in selected_users:
            # 预热判定：40轮后才开启压缩
            is_comp = False if epoch < 40 else (args.use_csmc and args.optimal_X[idx] == 0)
            if is_comp:
                comp_user_count += 1  # 统计压缩用户数

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
        if acc > best_acc:
            best_acc = acc

        # 新增：计算平均训练损失，对齐日志打印格式
        avg_train_loss = sum(round_train_losses) / len(round_train_losses) if round_train_losses else 0.0
        # 打印Loss日志（和之前版本一致）
        print(f'   [Loss] Train: {avg_train_loss:.4f} | Test: {test_loss:.4f}')
        # 打印Stat日志（新增压缩用户数统计）
        stat_log = f'   [Stat] Test Acc: {acc:.2%} | Best: {best_acc:.2%} | Comm: {total_bits / 1e6:.2f}Mb'
        if epoch >= 40:  # 40轮后才显示压缩用户数
            stat_log += f" | CompUsers: {comp_user_count}"
        print(stat_log)

        # ========== 新增：收敛条件判断核心逻辑 ==========
        # 1. 记录最近patience轮的测试损失
        recent_test_losses.append(test_loss)
        if len(recent_test_losses) > patience:
            recent_test_losses.pop(0)

        # 2. 判断是否满足“准确率达标 + 模型收敛”
        is_accurate = acc >= target_acc
        is_converged = False
        if len(recent_test_losses) == patience:
            loss_diff = max(recent_test_losses) - min(recent_test_losses)
            is_converged = loss_diff < convergence_threshold
            print(f'   [Convergence] Recent loss diff: {loss_diff:.6f} (converged: {is_converged})')

        # 3. 满足条件则停止训练
        if is_accurate and is_converged:
            print(f'\n>>> 满足停止条件：准确率达标({acc:.2%})且模型收敛，提前终止训练')
            stop_training = True

        # 每轮结束强制清理显存碎片
        torch.cuda.empty_cache()

    # 保存结果（对齐之前版本的文件名格式）
    save_dir = '../save/results/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 新增：文件名包含iid、eps、rho、compress_rank等关键参数，和之前版本一致
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    iid_tag = 'iid' if args.iid else 'noniid'
    file_name = f"{args.strategy}_{args.dataset}_{iid_tag}_eps{args.epsilon}_rho{args.rho}_rank{args.compress_rank}_{timestamp}.pth"
    # 保存结果（新增best_acc和total_rounds，方便后续分析）
    results = {
        'args': vars(args),
        'test_acc': test_acc_list,
        'cumulative_bits': cumulative_bits_list,
        'best_acc': best_acc,
        'total_rounds': len(test_acc_list)  # 实际训练轮数
    }
    torch.save(results, os.path.join(save_dir, file_name))
    print(f"\nTraining Finished. Max Acc: {max(test_acc_list):.2%} | Total Rounds: {len(test_acc_list)}")
    print(f"Results saved to: {os.path.join(save_dir, file_name)}")


if __name__ == '__main__':
    main()
