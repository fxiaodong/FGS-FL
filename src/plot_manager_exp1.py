#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# 解决多重加载 libiomp5md.dll 的冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 设置全局样式
plt.style.use('seaborn-v0_8-whitegrid')  # 使用更现代的网格样式
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


class PlotManagerExp1:
    def __init__(self, data_root='../save/results/exp1/', plot_dir='../save/plots/exp1/'):
        self.data_root = data_root
        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)
        # 加载数据
        self.non_iid_results = self._load_group('noniid')
        self.iid_results = self._load_group('iid')

    def _load_group(self, sub_folder):
        group_data = []
        path = os.path.join(self.data_root, sub_folder)
        if not os.path.exists(path):
            print(f"警告: 找不到目录 {path}")
            return group_data

        files = [f for f in os.listdir(path) if f.endswith('.pth')]
        for f in files:
            file_path = os.path.join(path, f)
            try:
                res = torch.load(file_path, map_location=torch.device('cpu'))
                # 统一标识符
                if 'FedLRP' in f:
                    res['label'] = 'FedLRP (Ours, R=64)'
                    res['color'] = '#D62728'
                    res['marker'] = 'o'
                elif 'dp_fedavg' in f:
                    res['label'] = 'FedAvg + DP (Baseline)'
                    res['color'] = '#7F7F7F'
                    res['marker'] = 's'
                else:
                    continue
                group_data.append(res)
            except Exception as e:
                print(f"解析 {f} 失败: {e}")
        return group_data

    def plot_convergence_curves(self):
        """图 1: 准确率 vs. 训练轮数 (1x2 子图)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # --- 左图: Non-IID (Exp 1.1 vs 1.3) ---
        self._plot_curve_on_ax(ax1, self.non_iid_results,
                               "Convergence in Non-IID Setting (epsilon=10)")

        # --- 右图: IID (Exp 1.2 vs 1.4) ---
        self._plot_curve_on_ax(ax2, self.iid_results,
                               "Convergence in IID Setting (epsilon=10)")

        plt.tight_layout()
        save_path = os.path.join(self.plot_dir, "exp1_accuracy_convergence.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"准确率收敛图已保存至: {save_path}")
        plt.show()

    def _plot_curve_on_ax(self, ax, results, title):
        for res in results:
            acc = [a * 100 if a <= 1.0 else a for a in res['test_acc']]
            rounds = range(1, len(acc) + 1)
            ax.plot(rounds, acc, label=res['label'], color=res['color'],
                    linewidth=2, marker=res['marker'], markevery=10, markersize=6)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Communication Rounds", fontsize=12)
        ax.set_ylabel("Test Accuracy (%)", fontsize=12)
        ax.set_ylim(0, 100)
        ax.legend(loc='lower right', frameon=True)

    def plot_communication_bar_chart(self):
        """图 2: 总通信量对比柱状图"""

        def get_total_comm(results):
            data = {'FedLRP': 0, 'Baseline': 0}
            for res in results:
                total_mb = res['cumulative_bits'][-1] / (1024 ** 2) if 'cumulative_bits' in res else 0
                if 'Ours' in res['label']:
                    data['FedLRP'] = total_mb
                else:
                    data['Baseline'] = total_mb
            return data

        non_iid_data = get_total_comm(self.non_iid_results)
        iid_data = get_total_comm(self.iid_results)

        labels = ['Non-IID ', 'IID ']
        ours_vals = [non_iid_data['FedLRP'], iid_data['FedLRP']]
        base_vals = [non_iid_data['Baseline'], iid_data['Baseline']]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 7))
        rects1 = ax.bar(x - width / 2, ours_vals, width, label='FedLRP (Ours, R=64)',
                        color='#D62728', edgecolor='black', alpha=0.8)
        rects2 = ax.bar(x + width / 2, base_vals, width, label='FedAvg + DP (Baseline)',
                        color='#7F7F7F', edgecolor='black', alpha=0.8)

        ax.set_ylabel('Total Communication Overhead (Mb)', fontsize=12, fontweight='bold')
        ax.set_title('Comparison of Total Communication Cost', fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
        ax.legend()

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        save_path = os.path.join(self.plot_dir, "exp1_communication_bar.png")
        plt.savefig(save_path, dpi=300)
        print(f"通信量柱状图已保存至: {save_path}")
        plt.show()


if __name__ == "__main__":
    pm = PlotManagerExp1()
    # 同时生成两张核心实验图
    pm.plot_convergence_curves()
    pm.plot_communication_bar_chart()