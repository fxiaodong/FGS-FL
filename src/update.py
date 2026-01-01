#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import numpy as np
import torch
import gc
from torch import nn
from torch.utils.data import DataLoader, Dataset
from fgs_utils import fgo_gradient_flattening, gsr_noise_injection, calculate_dp_sigma


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        image, label = self.dataset[self.idxs[index]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger=None):
        self.args = args
        self.idxs = list(idxs)
        self.device = torch.device(f'cuda:{args.gpu}' if args.gpu and args.gpu != 'None' else 'cpu')
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, self.idxs)
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        train_idxs = idxs[:int(0.8 * len(idxs))]
        val_idxs = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        test_idxs = idxs[int(0.9 * len(idxs)):]
        base_bs = self.args.local_bs
        if self.args.iid == 0: base_bs = min(base_bs, 32)
        local_bs = max(1, min(len(train_idxs), base_bs))
        trainloader = DataLoader(DatasetSplit(dataset, train_idxs), batch_size=local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, val_idxs), batch_size=max(1, len(val_idxs) // 10))
        testloader = DataLoader(DatasetSplit(dataset, test_idxs), batch_size=max(1, len(test_idxs) // 10))
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round, compress=False, epsilon=1.0, history=None):
        model.train()
        model.to(self.device)

        train_model = copy.deepcopy(model)
        train_model.train()

        optimizer = torch.optim.SGD(train_model.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        use_fgs = 'fgs' in self.args.strategy
        sigma = calculate_dp_sigma(epsilon) if epsilon > 0 else 0.0

        C_clip = 1.0
        rho = self.args.rho
        eta = self.args.lr
        epoch_loss = []

        # --- 策略 A: FGS-FL (严格复现 Algorithm 1) ---
        if self.args.strategy == 'fgs_fl':
            w_initial_params = [p.clone().detach() for p in model.parameters()]
            local_gradient_stream = []

            for ep in range(self.args.local_ep):
                batch_loss = []
                for images, labels in self.trainloader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    output = train_model(images)
                    loss = self.criterion(output, labels)
                    loss.backward()

                    raw_grad = torch.cat([p.grad.flatten() for p in train_model.parameters()])

                    total_norm = torch.norm(raw_grad, 2)
                    clip_coef = max(1.0, float(total_norm / C_clip))
                    clipped_grad = raw_grad / clip_coef

                    flat_grad = clipped_grad
                    if use_fgs and rho > 0:
                        original_params = [p.clone().detach() for p in train_model.parameters()]
                        perturbation = (rho * clipped_grad) / (torch.norm(clipped_grad, 2) + 1e-8)

                        ptr = 0
                        for p in train_model.parameters():
                            numel = p.numel()
                            p.data.add_(perturbation[ptr:ptr + numel].view(p.shape))
                            ptr += numel

                        optimizer.zero_grad()
                        self.criterion(train_model(images), labels).backward()
                        flat_grad = torch.cat([p.grad.flatten() for p in train_model.parameters()])

                        ptr = 0
                        for p, orig in zip(train_model.parameters(), original_params):
                            p.data.copy_(orig.data)

                    local_gradient_stream.append(flat_grad.detach().cpu())

                    noisy_sum_grad = gsr_noise_injection(
                        local_gradient_stream,
                        self.args.epochs,
                        m=10,
                        sigma=sigma,
                        C=C_clip,
                        device=self.device
                    )

                    cumulative_update = -eta * noisy_sum_grad
                    ptr = 0
                    for p_init, p_train in zip(w_initial_params, train_model.parameters()):
                        numel = p_init.numel()
                        update_k = cumulative_update[ptr:ptr + numel].view(p_init.shape)
                        p_train.data.copy_(p_init.data + update_k)
                        ptr += numel

                    batch_loss.append(loss.item())
                epoch_loss.append(np.mean(batch_loss))

            final_state = {k: v.cpu() for k, v in train_model.state_dict().items()}
            del local_gradient_stream, w_initial_params
            gc.collect()
            torch.cuda.empty_cache()
            return {'state': final_state, 'n_samples': len(self.idxs)}, np.mean(epoch_loss), None

        # --- 策略 B: 主实验 (增加数值稳定性保护) ---
        else:
            for ep in range(self.args.local_ep):
                batch_loss = []
                for images, labels in self.trainloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    output = train_model(images)
                    loss = self.criterion(output, labels)
                    if torch.isnan(loss): continue
                    loss.backward()

                    if use_fgs and self.args.rho > 0:
                        raw_grad = torch.cat([p.grad.flatten() for p in train_model.parameters()])
                        norm = torch.norm(raw_grad, p=2)
                        scale = torch.max(torch.tensor(1.0, device=self.device), norm / C_clip)
                        clipped_grad = raw_grad / scale
                        perturbation = self.args.rho * clipped_grad / (torch.norm(clipped_grad, p=2) + 1e-8)

                        ptr = 0
                        for p in train_model.parameters():
                            numel = p.numel()
                            p.data.add_(perturbation[ptr:ptr + numel].view(p.shape))
                            ptr += numel
                        optimizer.zero_grad()
                        self.criterion(train_model(images), labels).backward()
                        flat_grad = torch.cat([p.grad.flatten() for p in train_model.parameters()])
                        ptr = 0
                        for p in train_model.parameters():
                            numel = p.numel()
                            p.data.sub_(perturbation[ptr:ptr + numel].view(p.shape))
                            p.grad.data = flat_grad[ptr:ptr + numel].view(p.shape)
                            ptr += numel

                    torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=C_clip)
                    optimizer.step()
                    batch_loss.append(loss.item())
                epoch_loss.append(np.mean(batch_loss))

            upload_state = {}
            do_compression = compress and (self.args.compress_layers != 'None')
            initial_state = model.state_dict()
            final_state = train_model.state_dict()
            target_layers = self.args.compress_layers.split(',') if do_compression else []

            for name, W_target in final_state.items():
                W_init = initial_state[name].detach().cpu().float()
                W_final = W_target.detach().cpu().float()
                delta = W_final - W_init

                # 数值异常判定：防止 NaN/Inf 导致 SVD 崩溃
                if torch.isnan(delta).any() or torch.isinf(delta).any():
                    upload_state[name] = W_init  # 若爆炸则不更新
                    continue

                if any(lyr in name for lyr in target_layers) and name.endswith('.weight') and do_compression:
                    try:
                        # 核心修改：增加极小扰动增强矩阵秩稳定性
                        reg_delta = delta + torch.randn_like(delta) * 1e-9
                        U, s, Vh = torch.linalg.svd(reg_delta, full_matrices=False)

                        k = min(self.args.compress_rank, min(delta.shape))
                        w1 = U[:, :k] @ torch.diag(torch.sqrt(s[:k]))
                        w2 = torch.diag(torch.sqrt(s[:k])) @ Vh[:k, :]
                        if epsilon > 0:
                            C_target = 0.5
                            w1 = w1 * min(1.0, C_target / (torch.norm(w1, p=2) + 1e-8))
                            w2 = w2 * min(1.0, C_target / (torch.norm(w2, p=2) + 1e-8))
                            w1 += torch.randn_like(w1) * (sigma * C_target)
                            w2 += torch.randn_like(w2) * (sigma * C_target)
                        upload_state[name] = W_init + (w1 @ w2) * 0.5
                    except Exception as e:
                        # SVD 失败保底：采用全量 DP 逻辑，不压缩
                        print(f"Warning: SVD failed for {name}, falling back to non-compression. Error: {e}")
                        C_low = 0.5
                        scale = min(1.0, C_low / (torch.norm(delta, p=2) + 1e-8))
                        noisy_delta = (delta * scale) + torch.randn_like(delta) * (sigma * C_low)
                        upload_state[name] = W_init + noisy_delta

                elif name.endswith('.weight') and epsilon > 0:
                    C_low = 0.5
                    scale = min(1.0, C_low / (torch.norm(delta, p=2) + 1e-8))
                    noisy_delta = (delta * scale) + torch.randn_like(delta) * (sigma * C_low)
                    upload_state[name] = W_init + noisy_delta
                else:
                    upload_state[name] = W_final

            del train_model
            gc.collect()
            torch.cuda.empty_cache()
            return {'state': upload_state, 'n_samples': len(self.idxs)}, np.mean(epoch_loss), None


def test_inference(args, model, test_dataset):
    model.eval()
    device = torch.device(f'cuda:{args.gpu}' if args.gpu and args.gpu != 'None' else 'cpu')
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    criterion = nn.NLLLoss().to(device)
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total, total_loss / len(testloader)