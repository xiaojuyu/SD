from collections import OrderedDict
from ctypes import Union, cast
import math
from turtle import forward
from typing import Any, List
from torchvision.models.vgg import cfgs, VGG, make_layers, vgg11, vgg13, vgg16, vgg19
from .node import Neuron
import torch.nn as nn
import torch

__all__ = ['vgg7', 'vgg11', 'vgg13', 'vgg16', 'vgg19']

cfgs['CVGG7'] = [64, 64, 'M', 128, 128, 256, 'M']
cfgs['SVGG7'] = [64, 64, 'A', 128, 128, 256]
cfgs['SVGG19'] = [64, 64, 64, 64, 64, 64, 64, 64, 'A', 128, 128, 128, 128, 128, 128, 128, 128, 256]

class cnnSmallVGG(VGG):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 10,
    ) -> None:
        super(cnnSmallVGG, self).__init__(features, num_classes)
        self.classifier = nn.Sequential(
          nn.Linear(256 * 7 * 7, 1024),
          nn.ReLU(True),
          nn.Linear(1024, num_classes)
        )

def vgg7(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> cnnSmallVGG:
    return cnnSmallVGG(make_layers(cfgs['CVGG7'], batch_norm=False), **kwargs)


def make_neurons(cfg: List, in_channels, vth:int=1, leak:int=1) -> nn.Sequential:
    layers: List[nn.Module] = []
    down_num = 0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            down_num += 1
        elif v == 'A':
            layers += [Neuron(nn.AvgPool2d(kernel_size=2, stride=2), 0.75, leak, pool=True)]
            down_num += 1
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1, bias=False)
            layers += [Neuron(conv2d, vth, leak)]
            in_channels = v
    return nn.Sequential(*layers), down_num

class snnSmallVGG(nn.Module):
    def __init__(self,
        features:nn.Sequential,
        down_num:int,
        num_classes:int=10,
        HW:tuple=(42, 42),

        vth:int=1,
        leak:int=1,
        ) -> None:
        super(snnSmallVGG, self).__init__()
        
        self.vth = vth
        self.features = features
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        down_size = pow(2, 1+down_num)
        self.fc1 = Neuron(nn.Linear(256 * int(HW[0]/down_size)*int(HW[1]/down_size), 1024, bias=False), vth, leak)
        self.fc2 = nn.Linear(1024, num_classes, bias=False)


    def snn_forward(self, x):
        for m in self.features:
            x = m.snn_forward(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1.snn_forward(x)
        return x
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x/self.vth
    
    def updateScale(self, sign):
        for m in self.features:
            m.updateScale(sign)
        self.fc1.updateScale(sign)
    
    def reset_mem(self):
        for m in self.features:
            m.reset(0,0,0)
        self.fc1.reset(0,0,0)
    

def snnvgg7(vth, leak, num_classes, CHW)->snnSmallVGG:
    return snnSmallVGG(*make_neurons(cfgs['SVGG7'], CHW[0], vth, leak), 
                    num_classes=num_classes, HW=(CHW[1], CHW[2]), vth=vth, leak=leak)

def snnvgg19(vth, leak, num_classes, CHW)->snnSmallVGG:
    return snnSmallVGG(*make_neurons(cfgs['SVGG19'], CHW[0], vth, leak), 
                    num_classes, (CHW[1], CHW[2]), vth=vth, leak=leak)