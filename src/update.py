#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import numpy as np
import torch
import gc
from torch import nn
from torch.utils.data import DataLoader, Dataset
from fgs_utils import gsr_noise_injection, calculate_dp_sigma


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
        if self.args.iid == 0:
            base_bs = min(base_bs, 32)
        local_bs = max(1, min(len(train_idxs), base_bs))

        trainloader = DataLoader(DatasetSplit(dataset, train_idxs), batch_size=local_bs, shuffle=True, num_workers=0,
                                 pin_memory=True)
        validloader = DataLoader(DatasetSplit(dataset, val_idxs), batch_size=max(1, len(val_idxs) // 10), num_workers=0,
                                 pin_memory=True)
        testloader = DataLoader(DatasetSplit(dataset, test_idxs), batch_size=max(1, len(test_idxs) // 10),
                                num_workers=0, pin_memory=True)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round, compress=False, epsilon=1.0, history=None):
        model.train()
        model.to(self.device)
        train_model = copy.deepcopy(model)
        train_model.train()
        optimizer = torch.optim.SGD(train_model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        use_fgs = 'fgs' in self.args.strategy
        sigma = calculate_dp_sigma(epsilon) if epsilon > 0 else 0.0
        C_clip, rho, eta = 1.0, self.args.rho, self.args.lr
        epoch_loss = []

        if self.args.strategy == 'fgs_fl':
            w_initial_params = [p.clone().detach() for p in model.parameters()]
            local_gradient_stream = []
            for ep in range(self.args.local_ep):
                batch_loss = []
                for i, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    output = train_model(images)
                    loss = self.criterion(output, labels)
                    loss.backward()
                    raw_grad = torch.cat([p.grad.flatten().detach() for p in train_model.parameters()])
                    clipped_grad = raw_grad / max(1.0, float(torch.norm(raw_grad, 2) / C_clip))
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
                        flat_grad = torch.cat([p.grad.flatten().detach() for p in train_model.parameters()])
                        ptr = 0
                        for p, orig in zip(train_model.parameters(), original_params):
                            p.data.copy_(orig.data)
                        del original_params, perturbation

                    if i % 5 == 0:
                        local_gradient_stream.append(flat_grad.cpu())

                    if i % 20 == 0 or i == len(self.trainloader) - 1:
                        noisy_sum_grad = gsr_noise_injection(local_gradient_stream, self.args.epochs, 10, sigma, C_clip,
                                                             self.device)
                        cumulative_update = -eta * noisy_sum_grad.to(self.device)
                        ptr = 0
                        for p_init, p_train in zip(w_initial_params, train_model.parameters()):
                            numel = p_init.numel()
                            p_train.data.copy_(p_init.data + cumulative_update[ptr:ptr + numel].view(p_init.shape))
                            ptr += numel
                    batch_loss.append(loss.item())
                    if i % 20 == 0:
                        print(f"| User Train | Ep: {ep} Batch: {i}/{len(self.trainloader)} | Loss: {loss.item():.4f}")
                epoch_loss.append(np.mean(batch_loss))
                torch.cuda.empty_cache()
            final_state = {k: v.cpu() for k, v in train_model.state_dict().items()}
            del local_gradient_stream, w_initial_params, train_model
            gc.collect()
            torch.cuda.empty_cache()
            return {'state': final_state, 'n_samples': len(self.idxs)}, np.mean(epoch_loss), None

        else:
            for ep in range(self.args.local_ep):
                batch_loss = []
                for i, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    output = train_model(images)
                    loss = self.criterion(output, labels)
                    loss.backward()
                    if use_fgs and self.args.rho > 0:
                        raw_grad = torch.cat([p.grad.flatten().detach() for p in train_model.parameters()])
                        scale = torch.max(torch.tensor(1.0, device=self.device), torch.norm(raw_grad, 2) / C_clip)
                        clipped_grad = raw_grad / scale
                        perturbation = self.args.rho * clipped_grad / (torch.norm(clipped_grad, 2) + 1e-8)
                        ptr = 0
                        for p in train_model.parameters():
                            p.data.add_(perturbation[ptr:ptr + p.numel()].view(p.shape))
                            ptr += p.numel()
                        optimizer.zero_grad()
                        self.criterion(train_model(images), labels).backward()
                        flat_grad = torch.cat([p.grad.flatten().detach() for p in train_model.parameters()])
                        ptr = 0
                        for p in train_model.parameters():
                            numel = p.numel()
                            p.data.sub_(perturbation[ptr:ptr + numel].view(p.shape))
                            p.grad.data = flat_grad[ptr:ptr + numel].view(p.shape)
                            ptr += numel
                        del perturbation
                    torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=C_clip)
                    optimizer.step()
                    batch_loss.append(loss.item())
                    if i % 20 == 0:
                        print(
                            f"| User Train (Main) | Ep: {ep} Batch: {i}/{len(self.trainloader)} | Loss: {loss.item():.4f}")
                epoch_loss.append(np.mean(batch_loss))
                torch.cuda.empty_cache()

            initial_state = model.state_dict()
            final_state = train_model.state_dict()
            upload_state = {}
            target_layers = self.args.compress_layers.split(',') if (
                        compress and self.args.compress_layers != 'None') else []
            for name, W_target in final_state.items():
                W_init, W_final = initial_state[name].cpu().float(), W_target.cpu().float()
                delta = W_final - W_init
                if any(lyr in name for lyr in target_layers) and name.endswith('.weight'):
                    U, s, Vh = torch.linalg.svd(delta, full_matrices=False)
                    k = min(self.args.compress_rank, min(delta.shape))
                    w1, w2 = U[:, :k] @ torch.diag(torch.sqrt(s[:k])), torch.diag(torch.sqrt(s[:k])) @ Vh[:k, :]
                    if epsilon > 0:
                        w1 += torch.randn_like(w1) * (sigma * 0.5)
                        w2 += torch.randn_like(w2) * (sigma * 0.5)
                    upload_state[name] = W_init + (w1 @ w2)
                else:
                    upload_state[name] = W_final
            del train_model
            gc.collect()
            torch.cuda.empty_cache()
            return {'state': upload_state, 'n_samples': len(self.idxs)}, np.mean(epoch_loss), None


# --- 这里是补回的 test_inference 函数 ---
def test_inference(args, model, test_dataset):
    model.eval()
    device = torch.device(f'cuda:{args.gpu}' if args.gpu and args.gpu != 'None' else 'cpu')
    model.to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total, 0.0