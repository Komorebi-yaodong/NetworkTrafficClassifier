import torch.nn as nn
import torch
import numpy as np
from PIL import Image


class Config():
        def __init__(self) -> None:
            # 训练参数
            self.lr = 1e-6  # 学习率
            self.num_epochs = 100  # 训练代数
            self.require_improvement = 20000  # 进步要求最小batch值
            self.batch_size = 64
            self.weight_decay = 0.5
            self.device = torch.device('cpu')   # 设备
            self.model_name = "P2P_NeXt_de"
            self.save_path = "./save_ckpt/"+self.model_name+".ckpt"
            self.class_list = ["good","bad"]


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_channels, out_channels, cardinality, stride):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x

        out = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))
        out = nn.ReLU(inplace=True)(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        identity = self.shortcut(identity)

        out += identity
        out = nn.ReLU(inplace=True)(out)

        return out


class ResNeXt(nn.Module):
    def __init__(self, block, num_blocks, cardinality, num_classes=1000):
        super(ResNeXt, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(64) 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) 

        self.layer1 = self._make_layer(block, 64, num_blocks[0], cardinality)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], cardinality, 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], cardinality, 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], cardinality, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, cardinality, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, cardinality, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))
        x = self.maxpool(x) 
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x) 

        x = self.avgpool(x) 
        x = torch.flatten(x, 1)
        save_img(x)
        x = self.fc(x) 

        return x


def resnext50(num_classes=1000):
    return ResNeXt(Bottleneck, [3, 4, 6, 3], 32, num_classes)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3)
        )
        self.net = resnext50(num_classes=2)
        
    def forward(self,x):
        pre = nn.ReLU(inplace=True)(self.pre_layer(x))
        out = self.net(pre)
        return out


def save_img(x):
    img_data = x.reshape(32,32)
    # 获取指定通道的特征图，并转换为0-255之间的整数值
    normalized_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    image_data = (normalized_data.data.numpy() * 255).astype(np.uint8)
    # 转换为PIL图像对象
    image = Image.fromarray(image_data)
    resized_image = image.resize((512, 512), Image.BICUBIC)
    resized_image.save("./cache/showjpg.jpg")