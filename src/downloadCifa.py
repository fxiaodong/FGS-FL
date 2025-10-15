import torchvision


# 自动下载数据集
# train_set = torchvision.datasets.CIFAR10(root="../data/cifar/", train=True, download=True)
# test_set = torchvision.datasets.CIFAR10(root="../data/cifar/", train=False, download=True)
# 自动下载 MNIST 数据集
mnist_train_set = torchvision.datasets.MNIST(root="../data/mnist/", train=True, download=True)
mnist_test_set = torchvision.datasets.MNIST(root="../data/mnist/", train=False, download=True)

# 自动下载 Fashion-MNIST 数据集
fmnist_train_set = torchvision.datasets.FashionMNIST(root="../data/fmnist/", train=True, download=True)
fmnist_test_set = torchvision.datasets.FashionMNIST(root="../data/fmnist/", train=False, download=True)
# print(test_set[0])  #测试集第一个数据
# print(test_set.classes)   #测试集的类型

# img,target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])  #输出测试集类型名称
# img.show()   #图片查看