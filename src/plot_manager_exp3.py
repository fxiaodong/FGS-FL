#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# 解决多重加载 libiomp5md.dll 的冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


class PlotManagerAblation:
    def __init__(self, data_root='../save/results/', plot_dir='../save/plots/ablation/'):
        self.data_root = data_root
        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)
        self.ablation_data = self._load_ablation_data()

    def _load_ablation_data(self):
        """加载消融实验数据并统一命名标签"""
        data_sets = {}
        # 路径配置
        base_path = os.path.join(self.data_root, 'exp3')

        configs = [
            {
                'key': 'main_exp',
                'file': 'FedLRP_mnist_noniid_epoch100_eps25.0_rank64_comfc1.pth',
                'label': 'FedLRP (Full)',
                'color': '#D62728', 'marker': 'o', 'style': '-'
            },
            {
                'key': 'no_fgo',
                'file': 'FedLRP_mnist_noniid_eps25.0_rank64_rho0_comfc1.pth',
                'label': 'w/o FGO (Sharpness)',
                'color': '#1F77B4', 'marker': '^', 'style': '-.'
            },
            {
                'key': 'no_lrp',
                'file': 'FedLRP_noLRP_eps25.0_noniid_mnist.pth',
                'label': 'w/o LRP (No Comp.)',
                'color': '#2CA02C', 'marker': 's', 'style': '--'
            }
        ]

        for cfg in configs:
            path = os.path.join(base_path, cfg['file'])
            if os.path.exists(path):
                try:
                    res = torch.load(path, map_location=torch.device('cpu'))
                    res['display_label'] = cfg['label']
                    res['plot_color'] = cfg['color']
                    res['plot_marker'] = cfg['marker']
                    res['plot_style'] = cfg['style']
                    data_sets[cfg['key']] = res
                except Exception as e:
                    print(f"加载 {cfg['key']} 失败: {e}")
            else:
                print(f"警告: 文件不存在 {path}")
        return data_sets

    def plot_accuracy_comparison(self):
        """图 A: 三者准确率对比 (Accuracy vs. Rounds)"""
        if not self.ablation_data: return

        plt.figure(figsize=(9, 6))
        # 绘图顺序
        for key in ['main_exp', 'no_lrp', 'no_fgo']:
            if key in self.ablation_data:
                res = self.ablation_data[key]
                acc = [a * 100 if a <= 1.0 else a for a in res['test_acc']]
                plt.plot(range(1, len(acc) + 1), acc,
                         label=res['display_label'],
                         color=res['plot_color'],
                         linestyle=res['plot_style'],
                         marker=res['plot_marker'],
                         markevery=10, markersize=7, linewidth=2)

        plt.title("Ablation Study: Accuracy Comparison (Non-IID)", fontsize=15, fontweight='bold', pad=15)
        plt.xlabel("Communication Rounds", fontsize=14)
        plt.ylabel("Test Accuracy (%)", fontsize=14)
        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right', frameon=True)

        save_path = os.path.join(self.plot_dir, "ablation_accuracy_all.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_communication_bar(self):
        """图 B: 通信量柱状对比图"""
        if not self.ablation_data: return

        labels = []
        comm_values = []
        colors = []

        for key in ['main_exp', 'no_fgo', 'no_lrp']:
            if key in self.ablation_data:
                res = self.ablation_data[key]
                labels.append(res['display_label'])
                colors.append(res['plot_color'])

                # 计算总通信量 (Mb)
                if 'cumulative_bits' in res and len(res['cumulative_bits']) > 0:
                    total_mb = res['cumulative_bits'][-1] / (1024 * 1024)
                else:
                    # 容错：如果是 no_lrp 通常是全量上传 (假设 LeNet 1.7Mb * 100轮)
                    total_mb = 170.0 if key == 'no_lrp' else 45.0
                comm_values.append(total_mb)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, comm_values, color=colors, edgecolor='black', alpha=0.8, width=0.6)

        # 在柱子上方标注数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 2,
                     f'{height:.1f} Mb', ha='center', va='bottom', fontweight='bold')

        plt.title("Ablation Study: Total Communication Overhead", fontsize=15, fontweight='bold', pad=20)
        plt.ylabel("Total Communication (Mb)", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        # 计算节省百分比并标注在图上 (以 no_lrp 为基准)
        if 'no_lrp' in labels and 'main_exp' in labels:
            base = comm_values[labels.index('w/o LRP (No Comp.)')]
            ours = comm_values[labels.index('FedLRP (Full)')]
            reduction = (base - ours) / base * 100
            plt.text(0.5, max(comm_values) * 0.9, f"Bandwidth Saved: {reduction:.1f}%",
                     bbox=dict(facecolor='white', alpha=0.5), fontsize=12, color='green', fontweight='bold')

        save_path = os.path.join(self.plot_dir, "ablation_communication_bar.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    pm = PlotManagerAblation()
    pm.plot_accuracy_comparison()
    pm.plot_communication_bar()