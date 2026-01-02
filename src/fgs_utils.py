import torch
import numpy as np

def gsr_noise_injection(history_flat_grads, total_epochs, m, sigma, C, device):
    """
    梯度流释放 (GSR) - 彻底迁移至 CPU 计算，保护 6GB 显存
    """
    r = len(history_flat_grads)
    if r == 0:
        return torch.zeros(1).to(device)

    # 全程在 CPU 上处理庞大的梯度矩阵，防止 GPU 显存爆发
    G = torch.stack([g.cpu() for g in history_flat_grads])

    # 生成下三角矩阵 W (在 CPU)
    W = torch.tril(torch.ones((r, r)))

    try:
        # 在 CPU 上进行 SVD 分解
        U, s, Vh = torch.linalg.svd(W, full_matrices=False)
        m_eff = min(m, r)
        R_mat = U[:, :m_eff]
        M_mat = torch.diag(s[:m_eff]) @ Vh[:m_eff, :]

        # 在 CPU 上完成矩阵乘法
        compressed = torch.matmul(M_mat, G)
        xi = torch.randn_like(compressed) * (sigma * C)

        # 重构
        reconstructed_sums = torch.matmul(R_mat, (compressed + xi))

        # 只将结果送回 GPU
        res = reconstructed_sums[-1].flatten().to(device)
        del G, W, U, s, Vh, compressed, xi, reconstructed_sums
        return res
    except Exception as e:
        print(f"GSR Logic failed: {e}, falling back.")
        return history_flat_grads[-1].to(device) + torch.randn_like(history_flat_grads[-1]).to(device) * (sigma * C)

def calculate_dp_sigma(epsilon, delta=1e-5, C=1.0):
    if epsilon <= 0: return 0.0
    return np.sqrt(2 * np.log(1.25 / delta)) / epsilon