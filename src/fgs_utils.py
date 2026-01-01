import copy
import torch
import numpy as np
import torch.nn.functional as F

def fgo_gradient_flattening(model, raw_grad, rho=0.05, C=1.0):
    """
    梯度平坦化 (FGO)
    论文对应: Section 4.1, Eq. 11 - Eq. 12
    """
    # 1. 梯度裁剪 (Eq. 10)
    grad_norm = torch.norm(raw_grad, p=2)
    scale = torch.max(torch.tensor(1.0, device=raw_grad.device), grad_norm / C)
    clipped_grad = raw_grad / scale

    # 2. 计算扰动并应用到临时模型 (Eq. 11)
    perturbed_model = copy.deepcopy(model)
    ptr = 0

    # 计算归一化的扰动方向 (稳定性增强: 1e-8)
    norm_clipped = torch.norm(clipped_grad, p=2) + 1e-8
    perturbation_step = (rho * clipped_grad) / norm_clipped

    for p in perturbed_model.parameters():
        numel = p.numel()
        p_perturbation = perturbation_step[ptr:ptr + numel].view(p.shape)
        p.data.add_(p_perturbation)  # w_hat = w + rho * (g / ||g||)
        ptr += numel

    return perturbed_model

def gsr_noise_injection(history_flat_grads, total_epochs, m, sigma, C, device):
    """
    梯度流释放 (GSR) - 显存优化版
    强制在 CPU 上进行大矩阵运算，避免 GPU OOM
    """
    r = len(history_flat_grads)
    if r == 0:
        return torch.zeros_like(history_flat_grads[0]).to(device)

    # 1. 确保所有历史梯度都在 CPU 上，并堆叠
    # 不再使用 .to(device)，因为 device 可能是 cuda
    G = torch.stack([g.detach().cpu() for g in history_flat_grads]) # Shape: [r, D]

    # 2. 定义工作负载矩阵 W (r x r 下三角)
    W = np.tril(np.ones((r, r)))

    # 3. 矩阵分解 (在 CPU 上使用 numpy)
    try:
        U, s, Vh = np.linalg.svd(W, full_matrices=False)
    except np.linalg.LinAlgError:
        # 降级处理：如果分解失败，退回到简单加噪
        return (G[-1] + torch.randn_like(G[-1]) * (sigma * C)).to(device)

    m_eff = min(m, r)

    # 构造分解矩阵 (保持在 CPU)
    R_mat = torch.from_numpy(U[:, :m_eff]).float()
    M_mat = torch.from_numpy(np.diag(s[:m_eff]) @ Vh[:m_eff, :]).float()

    # 4. 压缩与加噪 (在 CPU 上完成 [m x r] * [r x D])
    # 这里是显存压力最大的地方，CPU 内存通常比显存大，可以扛住
    compressed = torch.matmul(M_mat, G)
    xi = torch.randn_like(compressed) * (sigma * C)

    # 5. 重构累积和 (reconstructed_sums shape: [r, D])
    reconstructed_sums = torch.matmul(R_mat, (compressed + xi))

    # 6. 只把最后一行（当前轮更新）搬回 GPU
    return reconstructed_sums[-1].flatten().to(device)

def calculate_dp_sigma(epsilon, delta=1e-5, C=1.0):
    """
    计算高斯噪声标准差 (Eq. 5)
    """
    if epsilon <= 0: return 0.0
    return np.sqrt(2 * np.log(1.25 / delta)) / epsilon