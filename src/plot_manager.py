#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体（Arial 适合英文论文，SimHei 用于支持中文显示）
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class PlotManager:
    def __init__(self, data_dir='../save/results/', plot_dir='../save/plots/'):
        self.data_dir = data_dir
        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)
        self.results = self._load_data()

        # 定义核心对比组的显示名称，用于过滤和统一图例
        self.target_labels = [
            r'FGS-csmcFL (Proposed, $R=8$)',
            r'DP-FedAvg (Baseline)',
            r'FGS-csmcFL (No Perturbation, $\rho=0$)',
            r'FGS-csmcFL (Large Rank, $R=512$)',
            r'csmcFL (No Privacy, $\epsilon=1000$)'
        ]

    def _load_data(self):
        data = []
        if not os.path.exists(self.data_dir):
            print(f"错误: 找不到目录 {self.data_dir}")
            return data

        files = [f for f in os.listdir(self.data_dir) if f.endswith('.pth')]
        print(f"在目录中发现 {len(files)} 个结果文件。开始解析内容...")

        for f in files:
            path = os.path.join(self.data_dir, f)
            try:
                # weights_only=False 是为了兼容 PyTorch 2.6 加载 numpy 对象
                res = torch.load(path, map_location='cpu', weights_only=False)
                args = res['args']

                # --- 核心识别逻辑：根据参数特征重新定义显示名称 ---
                if args['strategy'] == 'dp_fedavg_comp':
                    args['display_name'] = r'DP-FedAvg (Baseline)'
                elif args['strategy'] == 'fgs_csmcFL':
                    if args['epsilon'] > 100:
                        args['display_name'] = r'csmcFL (No Privacy, $\epsilon=1000$)'
                    elif args['rho'] == 0:
                        args['display_name'] = r'FGS-csmcFL (No Perturbation, $\rho=0$)'
                    elif args.get('compress_rank', 0) > 100:
                        args['display_name'] = r'FGS-csmcFL (Large Rank, $R=512$)'
                    else:
                        args['display_name'] = r'FGS-csmcFL (Proposed, $R=8$)'
                else:
                    args['display_name'] = args['strategy']

                data.append(res)
            except Exception as e:
                print(f"加载文件 {f} 失败: {e}")
        return data

    def plot_main_comparison(self, dataset='mnist', target_eps=20.0):
        """绘制：准确率 vs 通信轮数"""
        if not self.results: return
        plt.figure(figsize=(10, 6))

        # 排序确保 Proposed 线在最前端，不被遮挡
        self.results.sort(key=lambda x: 'Proposed' in x['args']['display_name'], reverse=True)

        for res in self.results:
            args = res['args']
            label = args['display_name']

            # 只绘制匹配当前数据集、隐私预算，且在核心展示列表中的数据
            if args['dataset'] == dataset and label in self.target_labels:
                if args['epsilon'] == target_eps or args['epsilon'] > 100:
                    acc = res['test_acc']

                    # 样式设定
                    color = 'red' if 'Proposed' in label else None
                    linewidth = 3.0 if 'Proposed' in label else 1.8
                    linestyle = '-'
                    if 'Baseline' in label:
                        color = 'gray'
                        linestyle = '--'

                    plt.plot(range(1, len(acc) + 1), acc, label=label,
                             color=color, linewidth=linewidth, linestyle=linestyle)

        plt.title(f"Accuracy Performance on {dataset.upper()} ($\epsilon={target_eps}$)", fontsize=14)
        plt.xlabel("Communication Rounds", fontsize=12)
        plt.ylabel("Test Accuracy", fontsize=12)
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)

        save_path = os.path.join(self.plot_dir, f"convergence_comparison_{dataset}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"收敛曲线图已保存至: {save_path}")
        plt.show()

    def plot_communication_efficiency(self, dataset='mnist'):
        """绘制：准确率 vs 累计通信量 (GB)"""
        if not self.results: return
        plt.figure(figsize=(10, 6))

        for res in self.results:
            args = res['args']
            label = args['display_name']

            if args['dataset'] == dataset and label in self.target_labels:
                # 转换位到GB: bits / (8 * 1024^3)
                gb_data = np.array(res['cumulative_bits']) / (8 * 1024 * 1024 * 1024)

                # 样式设定
                color = 'red' if 'Proposed' in label else None
                linewidth = 3.0 if 'Proposed' in label else 1.8
                linestyle = '-'
                if 'Baseline' in label:
                    color = 'gray'
                    linestyle = '--'

                plt.plot(gb_data, res['test_acc'], label=label,
                         color=color, linewidth=linewidth, linestyle=linestyle)

        plt.title("Efficiency: Accuracy vs. Transmitted Data", fontsize=14)
        plt.xlabel("Communication Overhead (GB)", fontsize=12)
        plt.ylabel("Test Accuracy", fontsize=12)
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)

        save_path = os.path.join(self.plot_dir, "communication_efficiency_cleaned.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"通信效率图已保存至: {save_path}")
        plt.show()


if __name__ == "__main__":
    # 使用时请确认 data_dir 指向你存放那 5 个核心 .pth 文件的位置
    pm = PlotManager(data_dir='../save/results/', plot_dir='../save/plots/')

    if not pm.results:
        print("未加载到有效数据，请检查 data_dir 路径和文件名。")
    else:
        # 1. 绘制收敛曲线
        pm.plot_main_comparison(dataset='mnist', target_eps=20.0)
        # 2. 绘制通信效率图
        pm.plot_communication_efficiency(dataset='mnist')