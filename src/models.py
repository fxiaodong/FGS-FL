import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankFC(nn.Module):
    def __init__(self, in_features, out_features, rank, device='cpu'):
        super().__init__()
        self.rank = int(rank)
        self.W1 = nn.Parameter(torch.randn(out_features, self.rank, device=device) * 0.01)
        self.W2 = nn.Parameter(torch.randn(self.rank, in_features, device=device) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features, device=device))

    def forward(self, x):
        weight = torch.matmul(self.W1, self.W2)
        return F.linear(x, weight, self.bias)

def replace_fc_with_lowrank(model, layer_name: str, rank: int):
    parts = layer_name.split('.')
    module = model
    for p in parts[:-1]:
        module = getattr(module, p)
    orig = getattr(module, parts[-1])

    device = orig.weight.device
    W = orig.weight.data.detach()
    bias = orig.bias.data.detach() if orig.bias is not None else None

    try:
        W_cpu = W.detach().cpu().to(torch.float32)
        U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
        U, S, Vh = U.to(W.device), S.to(W.device), Vh.to(W.device)
        k = min(rank, S.shape[0])
        U_r = U[:, :k] * torch.sqrt(S[:k]).unsqueeze(0)
        V_r = torch.sqrt(S[:k]).unsqueeze(1) * Vh[:k, :]
        new_layer = LowRankFC(orig.in_features, orig.out_features, k, device=device)
        new_layer.W1.data, new_layer.W2.data = U_r, V_r
        if bias is not None: new_layer.bias.data = bias
    except Exception as e:
        new_layer = LowRankFC(orig.in_features, orig.out_features, rank, device=device)
        if bias is not None: new_layer.bias.data = bias

    setattr(module, parts[-1], new_layer)
    return model

# --- 架构定义 ---

class MLP(nn.Module):
    def __init__(self, dim_in=784, dim_hidden=200, dim_out=10):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNNMnist(nn.Module):
    def __init__(self, args):
        super().__init__()
        in_ch = getattr(args, 'num_channels', 1)
        self.conv1 = nn.Conv2d(in_ch, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10) # Added BN
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(20) # Added BN
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, getattr(args, 'num_classes', 10))
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNNFashion_Mnist(CNNMnist): pass

class CNNCifar(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6) # Added BN
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16) # Added BN
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, getattr(args, 'num_classes', 10))
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class LeNet5(nn.Module):
    def __init__(self, args):
        super(LeNet5, self).__init__()
        in_ch = getattr(args, 'num_channels', 1)
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32) # Added BN
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64) # Added BN
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, getattr(args, 'num_classes', 10))
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class AlexNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), # Added BN
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192), # Added BN
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384), # Added BN
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), # Added BN
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), # Added BN
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, getattr(args, 'num_classes', 10)),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def get_fc_layer(model, layer_name):
    # 修复：如果传入的是 "fc1,fc2"，只返回第一个有效的层用于类型检查或参数获取
    # 因为 calculate_fl_time 通常只需要获取层对象的类型或基本属性
    if ',' in layer_name:
        layer_name = layer_name.split(',')[0].strip()

    parts = layer_name.split('.')
    module = model
    for p in parts:
        if hasattr(module, p):
            module = getattr(module, p)
        else:
            raise AttributeError(f"Model has no layer: {layer_name}")
    return module

def get_fc_rank(layer):
    if isinstance(layer, torch.nn.Linear):
        return min(layer.in_features, layer.out_features)
    if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
        return min(layer.in_features, layer.out_features)
    if hasattr(layer, 'rank'):
        return layer.rank
    return 0