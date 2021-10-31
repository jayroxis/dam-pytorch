from __future__ import absolute_import
import math

import torch.nn as nn
from .dam import DAM_2d, DAM

# from .channel_selection import channel_selection


__all__ = ['resnet']

"""
preactivation resnet with bottleneck design.
"""

    
class Bottleneck(nn.Module):
    def __init__(self, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.dam1 = DAM_2d(cfg[1]) 
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.dam2 = DAM_2d(cfg[2])
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes, kernel_size=1, bias=False) # 4 is expansion
        self.downsample = downsample
        self.dam3 = DAM_2d(planes)
        self.relu = nn.ReLU(inplace=True)

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
    
class Downsample_bottleneck(nn.Module):
    def __init__(self, downsample=None):
        super(Downsample_bottleneck, self).__init__()
        self.downsample = downsample
#         self.dam3 = DAM_2d(planes)

    def forward(self, x):
        residual = self.downsample(x)
        return residual
    
    
class BottleneckLayer(nn.Module):
    def __init__(self, planes, blocks, in_dims, stride=1):
        super(BottleneckLayer, self).__init__()
        
        nblock = len(blocks)
        
        if in_dims[3]!=0:
            downsample = nn.Sequential(
                nn.Conv2d(planes, in_dims[3] , kernel_size=1, stride=stride, bias=False),
            )
        else:
            downsample = nn.Sequential(
                nn.Conv2d(planes, planes , kernel_size=1, stride=stride, bias=False),
            )
        
        if blocks[0] == 1:
            if in_dims[0] == 0:
                in_block = Bottleneck(planes = in_dims[3], cfg = [planes]+in_dims[1:3], stride = stride, downsample = downsample)
            else:
                in_block = Bottleneck(planes = in_dims[3], cfg = in_dims[0:3], stride = stride, downsample = downsample)
        else:
            in_block = Downsample_bottleneck(downsample = downsample)
        
        layers = []
        layers.append(in_block)
        
        last_nonzero = 0
        for i in range(1, nblock):
            next_in_dims = in_dims[3*i: 3*(i+1)]
    
            if next_in_dims[0] == 0:
                if last_nonzero > 0:
                    next_in_dims[0] = in_dims[3*last_nonzero]
                else:
                    next_in_dims[0] = planes
            
            if blocks[i] == 0:  # pruning out the entire block if block[i] is zero
                continue
            if i == nblock-1:
                out_dim = planes
            else:
                last_nonzero = i
                out_dim = in_dims[3*(i+1)]
            block = Bottleneck(planes = out_dim, cfg = next_in_dims)
            layers.append(block)

        self.layers = nn.Sequential(*layers)
        
        if last_nonzero > 0:
            self.nonzero_dim = in_dims[3*(last_nonzero+1)]
        else:
            self.nonzero_dim = planes
        
    def forward(self, x):
        return self.layers(x)

    

class prune_resnet(nn.Module):
    def __init__(self, depth=164, dataset='cifar10', cfg=None, n_layers_list = None):
        super(prune_resnet, self).__init__()
#         nlayer = 18 # number of blocks per bottleneck layer
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'

        nlayer = (depth - 2) // 9
        
        if cfg is None:
            # Construct config variable.
            cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [128, 64, 64], [256, 64, 64]*(n-1), [256]]
            cfg = [item for sub_list in cfg for item in sub_list]
        
        if n_layers_list is None:
            n_layers_list = [1] * nlayer
            
        n1 = n_layers_list[ : nlayer]
        n2 = n_layers_list[nlayer : 2*nlayer]
        n3 = n_layers_list[2*nlayer:]

        self.conv1 = nn.Conv2d(3, cfg[0], kernel_size=3, padding=1,
                               bias=False)
        self.dam_in = DAM_2d(cfg[0])
        last_nonzero = cfg[0]
        if cfg[3] != 0:
            last_nonzero = cfg[3]
        self.layer1 = BottleneckLayer(last_nonzero, n1, in_dims = cfg[0:3*nlayer])
        self.layer2 = BottleneckLayer(self.layer1.nonzero_dim, n2, in_dims = cfg[3*nlayer:6*nlayer], stride=2)
        self.layer3 = BottleneckLayer(self.layer2.nonzero_dim, n3, in_dims = cfg[6*nlayer:9*nlayer], stride=2)
        self.bn = nn.BatchNorm2d(self.layer3.nonzero_dim)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        
        fc_in_dim = int(4 * self.layer3.nonzero_dim)
        
        
        print(fc_in_dim,self.layer3.nonzero_dim, self.layer2.nonzero_dim, self.layer1.nonzero_dim)
        
        
        if dataset == 'cifar10':
            self.fc = nn.Linear(fc_in_dim, 10)
        elif dataset == 'cifar100':
            self.fc = nn.Linear(fc_in_dim, 100)

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