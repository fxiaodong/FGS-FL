#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# update.py：客户端本地更新与模型评估（FedAvg 核心逻辑）
# 该文件定义了 客户端本地训练逻辑 和 全局模型评估逻辑，是 FedAvg 算法的 “执行核心”。

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# DatasetSplit：客户端本地数据集封装
# 将 “全局数据集 + 客户端数据索引” 封装为 PyTorch Dataset，供客户端加载本地数据：
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset # 全局数据集
        self.idxs = [int(i) for i in idxs] # 客户端对应的样本索引

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]] # 提取客户端本地样本
        return torch.tensor(image), torch.tensor(label)

# 客户端本地训练类（对应论文 “ClientUpdate”）
# 实现论文中 “客户端本地多轮 SGD 训练” 的完整逻辑：
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        # 划分客户端本地的“训练集/验证集/测试集”（8:1:1）
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        #  损失函数（对应论文交叉熵损失）
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        #   # 按 8:1:1 分割客户端本地数据（训练/验证/测试）
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]
        # 构建 DataLoader
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        """客户端本地多轮训练，返回更新后的模型权重和平均损失"""
        model.train()
        epoch_loss = []

        # 初始化优化器（对应论文 SGD）
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        # 本地多轮训练（对应论文 E 轮本地 epoch）
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)# 计算本地损失
                loss.backward()# 反向传播
                optimizer.step()# 本地 SGD 更新

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))# 每 epoch 平均损失

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ 客户端本地评估（返回准确率和损失）
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            '''这段代码的功能是：
            模型推理：outputs = model(images) - 将输入图像数据传入模型进行前向传播，得到预测输出
            计算损失：batch_loss = self.criterion(outputs, labels) - 使用预定义的损失函数计算当前批次的预测损失
            累积损失：loss += batch_loss.item() - 将当前批次损失值累加到总损失中，用于后续的统计和优化'''
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            #  计算准确率
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

# 全局模型测试（对应论文 “测试集评估”）
def test_inference(args, model, test_dataset):
    """ 在全局测试集上评估聚合后的模型性能，返回准确率和损失.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        #  计算准确率
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
