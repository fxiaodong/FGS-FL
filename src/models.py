#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F

# 轻量多层感知机，用于验证 FedAvg 对简单模型的适配性
# MNIST 2NN（2 个隐藏层，每个 200 单元，ReLU 激活），参数规模约 19.9 万。
class MLP(nn.Module):

    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden) # 输入层→隐藏层（200单元）
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out) # 隐藏层→输出层（10类）
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1]) # 展平图像（28×28→784）
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

# CNNMnist（MNIST 卷积模型）
# MNIST-CNN（2 个 5×5 卷积层 + 2 个池化层 + 2 个全连接层），参数规模约 166 万。
class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5) # 5×5卷积（1→10通道）
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 5×5卷积（10→20通道）
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)# 展平后→50单元全连接
        self.fc2 = nn.Linear(50, args.num_classes)# 50→10类

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # 卷积→ReLU→2×2池化
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) # 展平（20通道×4×4=320）
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        # 第一层卷积模块：输入1通道（灰度图）→ 输出16通道，提取低阶特征（如边缘、纹理）
        self.layer1 = nn.Sequential(
            # Conv2d(输入通道数, 输出通道数, 卷积核大小,  padding)
            nn.Conv2d(1, 16, kernel_size=5, padding=2),# 1→16通道，5×5卷积核；padding=2确保卷积后尺寸不变（28×28）
            nn.BatchNorm2d(16),# 批归一化：加速训练收敛，缓解梯度消失
            nn.ReLU(),
            nn.MaxPool2d(2))# 2×2最大池化：尺寸减半（28×28→14×14），保留关键特征，减少计算量
        # 第二层卷积模块：输入16通道→输出32通道，提取高阶特征（如衣物局部结构）
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2), # 16→32通道，5×5卷积核；padding=2确保尺寸不变（14×14
            nn.BatchNorm2d(32), # 批归一化
            nn.ReLU(),
            nn.MaxPool2d(2))
        # 全连接层：将卷积提取的特征展平后，映射到10类（Fashion-MNIST的类别数）
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)  # 输入x（batch_size, 1, 28, 28）→ layer1后（batch_size, 16, 14, 14）
        out = self.layer2(out)  # layer2后（batch_size, 32, 7, 7）
        out = out.view(out.size(0), -1)  # 展平：(batch_size, 32, 7, 7) → (batch_size, 32×7×7) = (batch_size, 1568)
        out = self.fc(out)  # 全连接层：(batch_size, 1568) → (batch_size, 10)（10个类别得分）
        return out

# CNNCifar（CIFAR-10 卷积模型）
# 适配彩色图像（3 通道），结构与论文中 CIFAR 实验一致：
class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 3通道→6通道（5×5卷积）
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) # 6→16通道（5×5卷积）
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 展平（16×5×5=400→120）
        self.fc2 = nn.Linear(120, 84)  # 120→84
        self.fc3 = nn.Linear(84, args.num_classes) # 84→10类

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

"""通用型全卷积网络（All Convolutional Network），
核心特点是 “用卷积层替代传统全连接层”，
支持自适应输入尺寸（无需固定输入图像大小），
同时通过多层卷积与 dropout 增强特征提取能力和抗过拟合能力，
适用于 MNIST、CIFAR-10 等多类图像分类场景（需根据数据集调整input_size输入通道数）。"""
class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, *args, **kwargs):
        super(modelC, self).__init__()  # 原代码super(AllConvNet, self).__init__()为笔误，正确应为modelC
        # 第一组卷积：输入通道→96通道，尺寸逐步减半
        self.conv1 = nn.Conv2d(input_size, 96, 3,
                               padding=1)  # 输入通道input_size（如MNIST=1，CIFAR=3）→96通道，3×3卷积，padding=1（尺寸不变）
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)  # 96→96通道，3×3卷积（强化特征）
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)  # 96→96通道，stride=2（尺寸减半）

        # 第二组卷积：96→192通道，尺寸再次减半
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)  # 96→192通道（提升特征维度）
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)  # 192→192通道（强化特征）
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)  # 192→192通道，stride=2（尺寸再减半）

        # 第三组卷积：192→192通道，1×1卷积压缩维度
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)  # 192→192通道
        self.conv8 = nn.Conv2d(192, 192, 1)  # 1×1卷积（无padding，尺寸不变）：减少计算量，调整通道相关性

        # 分类卷积层：192通道→n_classes通道（替代全连接层）
        self.class_conv = nn.Conv2d(192, n_classes, 1)  # 1×1卷积：将特征通道映射为类别数


    def forward(self, x):
        # 第一组卷积+dropout：输入→96通道，尺寸减半
        x_drop = F.dropout(x, .2)  # 输入 dropout（概率0.2）：防止输入层过拟合
        conv1_out = F.relu(self.conv1(x_drop))  # conv1→ReLU激活
        conv2_out = F.relu(self.conv2(conv1_out))  # conv2→ReLU
        conv3_out = F.relu(self.conv3(conv2_out))  # conv3（stride=2）→ReLU
        conv3_out_drop = F.dropout(conv3_out, .5)  # dropout（概率0.5）：增强抗过拟合

        # 第二组卷积+dropout：96→192通道，尺寸再减半
        conv4_out = F.relu(self.conv4(conv3_out_drop))  # conv4→ReLU
        conv5_out = F.relu(self.conv5(conv4_out))  # conv5→ReLU
        conv6_out = F.relu(self.conv6(conv5_out))  # conv6（stride=2）→ReLU
        conv6_out_drop = F.dropout(conv6_out, .5)  # dropout（概率0.5）

        # 第三组卷积：192→192通道
        conv7_out = F.relu(self.conv7(conv6_out_drop))  # conv7→ReLU
        conv8_out = F.relu(self.conv8(conv7_out))  # conv8（1×1）→ReLU

        # 分类：192通道→n_classes通道，自适应池化统一尺寸
        class_out = F.relu(self.class_conv(conv8_out))  # class_conv（1×1）→ReLU：输出(batch_size, n_classes, H, W)
        pool_out = F.adaptive_avg_pool2d(class_out, 1)  # 自适应平均池化：将(H,W)缩为(1,1)，输出(batch_size, n_classes, 1, 1)
        pool_out.squeeze_(-1)  # 移除最后1个维度：(batch_size, n_classes, 1)
        pool_out.squeeze_(-1)  # 再移除1个维度：(batch_size, n_classes)（最终输出）
        return pool_out
