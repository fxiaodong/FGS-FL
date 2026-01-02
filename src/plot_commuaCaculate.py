import torch

# 替换为你实际生成的文件名
fgs_res = torch.load('../save/results/fgs_fl_mnist_...pth')
main_res = torch.load('../save/results/main_experiment_...pth')

fgs_comm = fgs_res['cumulative_bits'][-1]  # 取最后一轮的总通讯量
main_comm = main_res['cumulative_bits'][-1]

reduction = (fgs_comm - main_comm) / fgs_comm * 100

print(f"FGS-FL 总通讯量: {fgs_comm/1e6:.2f} Mb")
print(f"主实验 总通讯量: {main_comm/1e6:.2f} Mb")
print(f"结论：主实验比 FGS-FL 通讯量降低了: {reduction:.2f}%")