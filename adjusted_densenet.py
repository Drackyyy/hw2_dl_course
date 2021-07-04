'''
Author: your name
Date: 2021-04-20 12:10:05
LastEditTime: 2021-04-22 14:56:54
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Homework2_yangxiaocong/dl_2021_hw2/dl_2021_hw2/hw2_start_code/src/adjusted_densenet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        # replace 3x3 kernel with 1x3 and 3x1
        self.add_module('conv2a', nn.Conv2d(bn_size * growth_rate, bn_size * growth_rate,
                                           kernel_size=(1,3), stride=1, padding=(0,1), bias=False)),
        self.add_module('conv2b', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=(3,1), stride=1, padding=(1,0), bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseLayer_with_noise(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer_with_noise, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        # replace 3x3 kernel with 1x3 and 3x1
        self.add_module('conv2a', nn.Conv2d(bn_size * growth_rate, bn_size *growth_rate,
                                           kernel_size=(1,3), stride=1, padding=(0,1), bias=False)),
        self.add_module('conv2b', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=(3,1), stride=1, padding=(1,0), bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        ## adding noise to denselayer when training
        new_features = super(_DenseLayer_with_noise, self).forward(x)
        new_features = new_features + torch.normal(mean = torch.zeros_like(new_features),std = torch.full_like(new_features,0.005)).cuda()
        if self.drop_rate > 0 :
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _DenseBlock_with_noise(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock_with_noise, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer_with_noise(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=10):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        # plan A: first six layers are denselayers with noise
        for i, num_layers in enumerate(block_config):
            if i == 0:
                block = _DenseBlock_with_noise(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                if i != len(block_config) - 1:
                    trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                    self.features.add_module('transition%d' % (i + 1), trans)
                    num_features = num_features // 2
            else:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
                self.features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * growth_rate
                if i != len(block_config) - 1:
                    trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                    self.features.add_module('transition%d' % (i + 1), trans)
                    num_features = num_features // 2
                    
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def adjusted_densenet121(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    return model


def densenet169(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    return model


def densenet201(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    return model


def densenet161(**kwargs):
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), **kwargs)
    return model


if __name__ == '__main__':
    # 'DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161'
    # Example
    net = DenseNet()
    print(net)
