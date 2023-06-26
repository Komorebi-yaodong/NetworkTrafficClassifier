# 网络攻防技术大作业

—— 善意、恶意流量分类

> 作品展示说明：本作品通过两种技术训练了四个模型用以实现流量分类；
>
> 每次分类前可以选择 **明文推理** 或者 **密文推理**；
>
> 每次推理展示两种方案的推理结果；
>
> 由于**随机森林模型较大**，加载网页时可能会卡顿；
>
> 无法得知测试机器配置如何，深度学习分类器推理在**CPU**上进行，速率根据测试机器不同而有差异（一般是秒出）。

## 小组成员

> **XXX，XXX，……**

## 技术与模型

> 本项目采用机器学习（随机森林 500棵树）与深度学习（残差神经网络NeXt）实现了对网络流量的分类。

### 机器学习——随机森林

#### 明文分类模型：

##### 使用的特征

```
['Src Port', 'Flow IAT Min', 'Fwd IAT Min', 'Flow Act Min', 'Flow Act Max', 'Flow Act Sum', 'Flow IAT Max', 'Flow Act Mean', 'Bwd IAT Min', 'Fwd IAT Max', 'Flow IAT Std', 'Flow Duration(ms)', 'Flow IAT Mean', 'Fwd Pkts/s', 'Flow Pkts/s', 'Fwd IAT Mean', 'Bwd IAT Max', 'Bwd Pkts/s', 'Fwd IAT Std', 'Bwd IAT Mean', 'Flow Act Std', 'Bwd IAT Std', 'Flow Idle Min', 'Bwd Init Win Bytes', 'Fwd Pld Byte Max', 'Flow Idle Mean', 'Flow Idle Sum', 'Flow Idle Max', 'Bytes Ratio', 'Fwd Pld Bytes/ms', 'Flow Pld Byte Std', 'Fwd Pld Byte Std', 'Flow Pld Byte Mean', 'Flow Pld Bytes/ms', 'Fwd Pld Byte Sum', 'Pkts Ratio', 'Bwd Pld Bytes/ms', 'Fwd Init Win Bytes', 'Sub Flow Fwd Bytes', 'Flow Pld Byte Sum', 'Dst Port', 'Fwd Pld Byte Mean', 'Bwd Pld Byte Std', 'Bwd Pld Byte Mean', 'Sub Flow Bwd Bytes', 'Bwd Pld Byte Sum', 'Fwd Head Byte Mean', 'Flow Pld Byte Max', 'Flow Idle Std', 'Fwd Head Byte Std', 'Bwd Head Byte Mean', 'ACK Count', 'Bwd Head Byte Std', 'Sub Flow Fwd Pkts', 'Bwd Avg Bulk/s', 'Bwd Pld Byte Max', 'Sub Flow Bwd Pkts', 'Flow Pkt Num', 'Fwd Pkt Num', 'Bwd Avg Bytes/Bulk', 'FIN Count', 'Bwd Pkt Num', 'PSH Count', 'Bwd Pkts With Pld', 'Bwd Avg Pkts/Bulk', 'Bwd PSH Count', 'Bwd Head Byte Max', 'Fwd Head Byte Min', 'Fwd PSH Count', 'Fwd Pkts With Pld', 'RST Count', 'SYN Count', 'Fwd Head Byte Max', 'Bwd Head Byte Min']
```

##### 模型准确率

![kquzeo.png](https://files.catbox.moe/kquzeo.png)

##### 参数重要性排序

1) Src Port                       0.050112
2) Flow IAT Min                   0.039806
3) Fwd IAT Min                    0.036455
4) Flow Act Min                   0.029925
5) Flow Act Max                   0.028885
6) Flow IAT Max                   0.028117
7) Flow Act Mean                  0.028107
8) Flow Act Sum                   0.028074
9) Bwd IAT Min                    0.027454
10) Fwd IAT Max                    0.025401
11) Flow Duration(ms)              0.024939
12) Flow IAT Std                   0.024935
13) Fwd Pkts/s                     0.023496
14) Flow IAT Mean                  0.023485
15) Flow Pkts/s                    0.022897
16) Fwd IAT Mean                   0.022817
17) Bwd IAT Max                    0.021359
18) Fwd IAT Std                    0.021231
19) Bwd Pkts/s                     0.021226
20) Bwd IAT Mean                   0.019475
21) Flow Act Std                   0.017157
22) Bwd IAT Std                    0.015753
23) Flow Idle Min                  0.015624
24) Bwd Init Win Bytes             0.014943
25) Fwd Pld Byte Max               0.014555
26) Bytes Ratio                    0.014119
27) Flow Idle Mean                 0.014110
28) Flow Idle Sum                  0.014021
29) Flow Idle Max                  0.013924
30) Fwd Pld Bytes/ms               0.013670
31) Flow Pld Byte Std              0.013494
32) Flow Pld Byte Mean             0.013357
33) Fwd Pld Byte Std               0.013281
34) Fwd Pld Byte Sum               0.013230
35) Flow Pld Bytes/ms              0.013063
36) Pkts Ratio                     0.013036
37) Bwd Pld Bytes/ms               0.012442
38) Sub Flow Fwd Bytes             0.012035
39) Flow Pld Byte Sum              0.011898
40) Dst Port                       0.011891
41) Fwd Init Win Bytes             0.011792
42) Fwd Pld Byte Mean              0.011652
43) Bwd Pld Byte Std               0.010270
44) Bwd Pld Byte Mean              0.009542
45) Bwd Pld Byte Sum               0.009261
46) Sub Flow Bwd Bytes             0.009189
47) Fwd Head Byte Mean             0.008972
48) Flow Pld Byte Max              0.007374
49) Flow Idle Std                  0.006925
50) Bwd Head Byte Mean             0.006823
51) Fwd Head Byte Std              0.006781
52) Sub Flow Fwd Pkts              0.005917
53) ACK Count                      0.005888
54) Bwd Head Byte Std              0.005854
55) Bwd Avg Bulk/s                 0.005543
56) Bwd Pld Byte Max               0.005185
57) Sub Flow Bwd Pkts              0.005181
58) Flow Pkt Num                   0.005038
59) Fwd Pkt Num                    0.004300
60) Bwd Avg Bytes/Bulk             0.004140
61) FIN Count                      0.003611
62) Bwd Pkt Num                    0.003459
63) PSH Count                      0.002916
64) Bwd Pkts With Pld              0.002695
65) Bwd Head Byte Max              0.002488
66) Bwd PSH Count                  0.002455
67) Bwd Avg Pkts/Bulk              0.002424
68) Fwd Head Byte Min              0.002076
69) Fwd Pkts With Pld              0.001630
70) Fwd PSH Count                  0.001525
71) RST Count                      0.001493
72) SYN Count                      0.001473
73) Fwd Head Byte Max              0.001300
74) Bwd Head Byte Min              0.001025

#### 密文分类模型：

##### 使用的特征

```
['Src Port', 'Flow IAT Min', 'Fwd IAT Min', 'Flow Act Min', 'Flow IAT Max', 'Flow Act Mean', 'Flow Act Max', 'Flow Act Sum', 'Flow Duration(ms)', 'Fwd Pkts/s', 'Dst Port', 'Flow Pkts/s', 'Flow IAT Mean', 'Bwd Pkts/s', 'Fwd IAT Max', 'Flow IAT Std', 'Fwd Init Win Bytes', 'Bwd Init Win Bytes', 'Bwd IAT Min', 'Fwd IAT Mean', 'Bwd IAT Max', 'Bwd IAT Mean', 'Fwd IAT Std', 'Flow Pld Bytes/ms', 'Fwd Pld Bytes/ms', 'Bwd IAT Std', 'Fwd Pld Byte Std', 'Flow Pld Byte Std', 'Fwd Pld Byte Max', 'Bytes Ratio', 'Flow Pld Byte Mean', 'Bwd Pld Bytes/ms', 'Fwd Pld Byte Sum', 'Flow Pld Byte Sum', 'Sub Flow Fwd Bytes', 'Fwd Pld Byte Mean', 'Pkts Ratio', 'Bwd Pld Byte Std', 'Flow Idle Min', 'Sub Flow Bwd Bytes', 'Bwd Pld Byte Mean', 'Flow Idle Sum', 'Bwd Pld Byte Sum', 'Flow Idle Mean', 'Flow Idle Max', 'Flow Act Std', 'Bwd Head Byte Mean', 'Fwd Head Byte Mean', 'Bwd Avg Bulk/s', 'Fwd Head Byte Std', 'ACK Count', 'Sub Flow Fwd Pkts', 'Flow Pkt Num', 'Bwd Avg Bytes/Bulk', 'Sub Flow Bwd Pkts', 'Bwd Head Byte Std', 'Fwd Pkt Num', 'PSH Count', 'Flow Pld Byte Max', 'Bwd Pkt Num', 'Bwd PSH Count', 'Flow Idle Std', 'Bwd Pkts With Pld', 'Bwd Pld Byte Max', 'Fwd Pkts With Pld', 'Fwd PSH Count', 'FIN Count', 'Bwd Avg Pkts/Bulk', 'Fwd Avg Bulk/s', 'Fwd Avg Bytes/Bulk', 'Bwd Head Byte Max', 'RST Count', 'Bwd Head Byte Min', 'Fwd Avg Pkts/Bulk', 'Fwd Head Byte Max', 'Fwd Head Byte Min', 'Fwd Pld Byte Min', 'SYN Count', 'Bwd Pld Byte Min', 'Flow Pld Byte Min']
```

##### 模型准确率

![e8gklv.png](https://files.catbox.moe/e8gklv.png)

##### 参数重要性排序

1) Src Port                       0.077231
2) Fwd IAT Min                    0.031252
3) Flow IAT Min                   0.031159
4) Flow Act Min                   0.027102
5) Flow IAT Max                   0.026225
6) Flow Act Mean                  0.025141
7) Flow Act Max                   0.025038
8) Flow Act Sum                   0.024719
9) Fwd Pkts/s                     0.024365
10) Flow Duration(ms)              0.024119
11) Dst Port                       0.023672
12) Flow Pkts/s                    0.023457
13) Flow IAT Mean                  0.023061
14) Bwd Pkts/s                     0.022595
15) Fwd IAT Max                    0.021408
16) Flow IAT Std                   0.020594
17) Fwd Init Win Bytes             0.020231
18) Bwd Init Win Bytes             0.020216
19) Bwd IAT Min                    0.020074
20) Fwd IAT Mean                   0.020053
21) Bwd IAT Max                    0.018393
22) Bwd IAT Mean                   0.017342
23) Fwd IAT Std                    0.016770
24) Flow Pld Bytes/ms              0.015490
25) Fwd Pld Bytes/ms               0.015418
26) Fwd Pld Byte Std               0.013955
27) Bwd IAT Std                    0.013834
28) Fwd Pld Byte Max               0.013208
29) Flow Pld Byte Std              0.013202
30) Bytes Ratio                    0.012993
31) Bwd Pld Bytes/ms               0.012688
32) Fwd Pld Byte Sum               0.012655
33) Flow Pld Byte Mean             0.012566
34) Flow Pld Byte Sum              0.012373
35) Sub Flow Fwd Bytes             0.012189
36) Fwd Pld Byte Mean              0.011971
37) Pkts Ratio                     0.011831
38) Flow Idle Min                  0.011306
39) Bwd Pld Byte Std               0.011292
40) Bwd Pld Byte Mean              0.010593
41) Sub Flow Bwd Bytes             0.010544
42) Flow Idle Mean                 0.010521
43) Flow Idle Sum                  0.010509
44) Flow Idle Max                  0.010448
45) Bwd Pld Byte Sum               0.010309
46) Flow Act Std                   0.009905
47) Bwd Head Byte Mean             0.008032
48) Fwd Head Byte Mean             0.007840
49) Bwd Avg Bulk/s                 0.007375
50) Fwd Head Byte Std              0.007375
51) ACK Count                      0.006763
52) Flow Pkt Num                   0.006435
53) Sub Flow Fwd Pkts              0.006312
54) Bwd Avg Bytes/Bulk             0.006083
55) Sub Flow Bwd Pkts              0.005982
56) Bwd Head Byte Std              0.005867
57) Fwd Pkt Num                    0.005795
58) PSH Count                      0.005207
59) Flow Pld Byte Max              0.004882
60) Bwd PSH Count                  0.004616
61) Bwd Pkt Num                    0.004597
62) Flow Idle Std                  0.004385
63) Bwd Pkts With Pld              0.004098
64) Bwd Pld Byte Max               0.003830
65) Fwd PSH Count                  0.003634
66) Fwd Pkts With Pld              0.003629
67) FIN Count                      0.003460
68) Bwd Avg Pkts/Bulk              0.003183
69) Fwd Avg Bulk/s                 0.002706
70) Fwd Avg Bytes/Bulk             0.002640
71) Bwd Head Byte Max              0.001979
72) RST Count                      0.001749
73) Bwd Head Byte Min              0.001677
74) Fwd Avg Pkts/Bulk              0.001348
75) Fwd Head Byte Max              0.001310
76) Fwd Head Byte Min              0.001258
77) Fwd Pld Byte Min               0.000692
78) SYN Count                      0.000640
79) Bwd Pld Byte Min               0.000305
80) Flow Pld Byte Min              0.000302

### 深度学习——残差网络RESNET

#### 明文分类模型

```
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
        x = x.squeeze(1)
        pre = nn.ReLU(inplace=True)(self.pre_layer(x))
        out = self.net(pre)
        return out
```

##### 模型准确率

![q7yg6o.png](https://files.catbox.moe/q7yg6o.png)

#### 密文分类模型

```
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
        x = x.squeeze(1)
        pre = nn.ReLU(inplace=True)(self.pre_layer(x))
        out = self.net(pre)
        return out
```

##### 模型准确率

![ql8lah.png](https://files.catbox.moe/ql8lah.png)

## 其他

### 1. 训练记录图

由于训练耗费大量时间，且中间伴随多次微调，导致多数训练记录不完整，取其中比较完整的一次作为展示（如图为NeXt网络针对密文分类时的训练记录）

![21m9di.png](https://files.catbox.moe/21m9di.png)

### 错误处理

由于随机森林模型较大，需要较大的运行内存，在训练与测试时曾出现过虚拟内存不足，导致模型再次运行时无法正确推理，可以主动调整虚拟内存大小以解决（本项目中使用的虚拟内存大小为128G），但是可能导致无法重现模型，目前还未找到解决方法，深度学习不存在此类问题。

### 继续训练

由于时间有限，以上四个模型可以继续训练，但准确率提高空间不大。
