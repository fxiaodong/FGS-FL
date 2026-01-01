#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# 解决多重加载 libiomp5md.dll 的冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体（Arial 适合英文论文）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


class PlotManager:
    def __init__(self, data_root='../save/results/', plot_root='../save/plots/'):
        self.data_root = data_root
        self.plot_root = plot_root
        os.makedirs(plot_root, exist_ok=True)

    def _load_files(self, sub_path):
        """通用加载函数，兼容旧版 PyTorch"""
        data = []
        full_path = os.path.join(self.data_root, sub_path)
        if not os.path.exists(full_path):
            print(f"警告: 找不到目录 {full_path}")
            return data

        files = [f for f in os.listdir(full_path) if f.endswith('.pth')]
        for f in files:
            try:
                res = torch.load(os.path.join(full_path, f), map_location=torch.device('cpu'))
                res['filename'] = f  # 记录文件名用于后续识别
                data.append(res)
            except Exception as e:
                print(f"解析 {f} 失败: {e}")
        return data

    # ================= 实验 1: 通信效率对比 (Acc vs. Comm) =================
    def plot_exp1_communication_efficiency(self):
        """生成 1x2 子图：对比 FedLRP 与 FedAvg+DP 的通信效率"""
        iid_data = self._load_files('exp1/iid')
        non_iid_data = self._load_files('exp1/non_iid')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        self._plot_comm_ax(ax1, non_iid_data, "Exp 1.1 vs 1.3: Non-IID Efficiency")
        self._plot_comm_ax(ax2, iid_data, "Exp 1.2 vs 1.4: IID Efficiency")

        plt.tight_layout()
        save_path = os.path.join(self.plot_root, "exp1_comm_efficiency_comparison.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"实验 1 通信效率图已保存至: {save_path}")
        plt.show()

    def _plot_comm_ax(self, ax, results, title):
        for res in results:
            f = res['filename']
            # 识别算法类型
            if 'FedLRP' in f:
                label = 'FedLRP (Ours, R=64)'
                color = '#D62728'
                style = '-'
            elif 'dp_fedavg' in f:
                label = 'FedAvg + DP (No Compression)'
                color = '#7F7F7F'
                style = '--'
            else:
                continue

            # 转换累计通信量为 Mb (1 Mb = 1024 * 1024 bits)
            # 注意：请确保你的 .pth 包含 'cumulative_bits' 字段
            comm_mb = np.array(res.get('cumulative_bits', [])) / (1024 * 1024)
            acc = [a * 100 if a <= 1.0 else a for a in res['test_acc']]

            # 如果 bits 记录不全，根据轮数和模型大小进行估算（ LeNet 约为 1.7Mb 全量）
            if len(comm_mb) == 0:
                step = 1.7 if 'dp_fedavg' in f else 0.45  # 0.45 是 R=64 的估算值
                comm_mb = np.cumsum([step] * len(acc))

            ax.plot(comm_mb, acc, label=label, color=color, linestyle=style, linewidth=2.5)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Cumulative Communication (Mb)", fontsize=12)
        ax.set_ylabel("Test Accuracy (%)", fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='lower right', fontsize=10)

    # ================= 实验 2: 隐私预算对比 (Acc vs. Rounds) =================
    def plot_exp2_privacy_comparison(self):
        """生成 1x2 子图：对比不同 epsilon 下的性能"""
        iid_data = self._load_files('exp2/iid')
        non_iid_data = self._load_files('exp2/non_iid')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        self._plot_round_ax(ax1, non_iid_data, "Exp 2.1 & 2.3: Non-IID Privacy Analysis")
        self._plot_round_ax(ax2, iid_data, "Exp 2.2 & 2.4: IID Privacy Analysis")

        plt.tight_layout()
        save_path = os.path.join(self.plot_root, "exp2_privacy_comparison.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"实验 2 隐私对比图已保存至: {save_path}")
        plt.show()

    def _plot_round_ax(self, ax, results, title):
        # 排序确保 eps=25 (低噪) 线在前面
        results.sort(key=lambda x: 'eps25.0' in x['filename'], reverse=True)

        for res in results:
            f = res['filename']
            if 'eps25.0' in f:
                label = r'$\epsilon=25$ (Low Noise)'
                color = '#1F77B4'
                style = '-'
            elif 'eps5.0' in f:
                label = r'$\epsilon=5$ (High Noise)'
                color = '#D62728'
                style = '--'
            else:
                continue

            acc = [a * 100 if a <= 1.0 else a for a in res['test_acc']]
            rounds = range(1, len(acc) + 1)
            ax.plot(rounds, acc, label=label, color=color, linestyle=style, linewidth=2.5)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Communication Rounds", fontsize=12)
        ax.set_ylabel("Test Accuracy (%)", fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='lower right', fontsize=11)


if __name__ == "__main__":
    pm = PlotManager()

    # 1. 绘制实验 1: 通信效率对比图
    print("正在生成实验 1 对比图...")
    pm.plot_exp1_communication_efficiency()

    # 2. 绘制实验 2: 隐私预算对比图
    print("正在生成实验 2 对比图...")
    pm.plot_exp2_privacy_comparison()