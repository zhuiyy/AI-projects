# 第二课作业
# 用pytorch实现卷积神经网络，对cifar10数据集进行分类
# 要求:1. 使用pytorch的nn.Module和Conv2d等相关的API实现卷积神经网络
#      2. 使用pytorch的DataLoader和Dataset等相关的API实现数据集的加载
#      3. 修改网络结构和参数，观察训练效果
#      4. 使用数据增强，提高模型的泛化能力

import os
import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm


# 定义超参数
batch_size = 256
learning_rate = 0.001
num_epochs = 100

# 定义数据预处理方式
# 普通的数据预处理方式
# transform = transforms.Compose([transforms.ToTensor(),])
# 数据增强的数据预处理方式
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])



# 定义数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.CAseries = [F.relu, F.relu, F.relu]
        self.LAseries = [F.relu, F.relu]

        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.Conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv_length = len(self.CAseries)
        self.linear_length = len(self.LAseries)

        self.cdropout = nn.Dropout2d(0.1)
        self.ldropout = nn.Dropout(0.5)
    def forward(self, input):
        input = self.Conv1(input)
        input = self.bn1(input)
        input = self.CAseries[0](input)
        input = F.max_pool2d(input, 2)
        input = self.cdropout(input)

        input = self.Conv2(input)
        input = self.bn2(input)
        input = self.CAseries[1](input)
        input = F.max_pool2d(input, 2)
        input = self.cdropout(input)

        input = self.Conv3(input)
        input = self.bn3(input)
        input = self.CAseries[2](input)
        input = F.max_pool2d(input, 2)
        input = self.cdropout(input)

        input = input.view(input.size(0), -1)

        input = self.fc1(input)
        input = self.LAseries[0](input)
        input = self.ldropout(input)

        input = self.fc2(input)
        input = self.LAseries[0](input)
        input = self.ldropout(input)

        input = self.fc3(input)

        return input
# 实例化模型
model = Net()

use_mlu = False
try:
    use_mlu = torch.mlu.is_available()
except:
    use_mlu = False

if use_mlu:
    device = torch.device('mlu:0')
    print('using MLU...')
else:
    print("MLU is not available, use GPU/CPU instead.")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('using GPU...')
    else:
        device = torch.device('cpu')
        print('using CPU...')

model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# 训练模型
for epoch in range(num_epochs):
    # 训练模式
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = (outputs.argmax(1) == labels).float().mean()

        # 打印训练信息
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), accuracy.item() * 100))

    # 测试模式
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Test Accuracy of the model on the 10000 test images: {100 * correct / total} %')
    scheduler.step()