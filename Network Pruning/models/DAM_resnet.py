import math
import torch
import torch.nn as nn
from .dam import DAM_2d, DAM
""" 
Code taken and modified from https://github.com/Eric-mingjie/network-slimming/blob/master/models/preresnet.py
"""
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, planes, cfg, gate_type, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.dam1 = DAM_2d(cfg[1], gate_type = gate_type)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.dam2 = DAM_2d(cfg[2], gate_type = gate_type)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes, kernel_size=1, bias=False)
        self.dam3 = DAM_2d(planes, gate_type = gate_type)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dam1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dam2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.dam3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        return out

class BottleneckLayer(nn.Module):
    def __init__(self, planes, blocks, cfg, gate_type, stride=1):
        super(BottleneckLayer, self).__init__()
        downsample = nn.Sequential(
            nn.Conv2d(cfg[0], planes, kernel_size=1, stride=stride, bias=False),
        )
        in_block = Bottleneck(planes = cfg[3], cfg = cfg[0:3], stride = stride, downsample = downsample, gate_type = gate_type)
        layers = []
        layers.append(in_block)
        
        for i in range(1, blocks):
            if i == blocks-1:
                out_dim = planes
            else:
                out_dim = cfg[3*(i+1)]
            block = Bottleneck(planes = out_dim, cfg = cfg[3*i: 3*(i+1)], gate_type = gate_type)
            layers.append(block)

        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


class resnet(nn.Module):
    def __init__(self, num_classes, depth=164, cfg=None, gate_type = 'relu_tanh'):
        super(resnet, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'

        n = (depth - 2) // 9

        if cfg is None:
            # Construct config variable.
            cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [128, 64, 64], [256, 64, 64]*(n-1), [256]]
            cfg = [item for sub_list in cfg for item in sub_list]


        self.conv1 = nn.Conv2d(3, cfg[0], kernel_size=3, padding=1,
                               bias=False)
        self.dam_in = DAM_2d(cfg[0], gate_type = gate_type)
        self.layer1 = BottleneckLayer(cfg[3*n], n, cfg = cfg[0:3*n], gate_type = gate_type)
        self.layer2 = BottleneckLayer(cfg[6*n], n, cfg = cfg[3*n:6*n], stride=2, gate_type = gate_type)
        self.layer3 = BottleneckLayer(cfg[9*n], n, cfg = cfg[6*n:9*n], stride=2, gate_type = gate_type)
        
        self.bn = nn.BatchNorm2d(cfg[9*n])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        self.fc = nn.Linear(cfg[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.dam_in(x)
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def get_DAM_model(method, num_classes, depth):
    net = resnet(num_classes, depth)
    return net