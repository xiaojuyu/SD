from collections import OrderedDict
from re import L
from typing import Any, List
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101
import torch.nn as nn
import torch

from event_datasets.network.node import Neuron

cfgs = {'resnet9': [
        ('backbone', [64, 'M']),
        ('block', [64, 128]),
        ('block', [128, 256]),
        ('block', [256, 512]),
    ]}

def make_layers(cfg) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for m in cfg:
        if m[0] is 'backbone':
            for v in m[1]:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1, bias=False)
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    in_channels = v
        elif m[0] is 'block':
            layers += [cnnBasicBlock(m[1][0], m[1][1])]
    return nn.Sequential(*layers)

class cnnBasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        outplanes: int,
    ) -> None:
        super(cnnBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=2, padding=0, bias=False)
        self.bndown = nn.BatchNorm2d(outplanes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(x)
        identity = self.bndown(identity)

        out += identity
        out = self.relu(out)

        return out

class cnnSmallResNet(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 10,
    ) -> None:
        super(cnnSmallResNet, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(OrderedDict([
          ('fc1', nn.Linear(512*7*7, 1024)),
          ('fc2', nn.Linear(1024, num_classes)),
            ])
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def resnet9(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> cnnSmallResNet:
    return cnnSmallResNet(make_layers(cfgs['resnet9']), **kwargs)




class ResponseFunc_skip(nn.Module):
    def __init__(self, cnn, skip) -> None:
        super(ResponseFunc_skip, self).__init__()
        self.cnn = cnn
        self.skip = skip
    def forward(self, x):
        return self.cnn(x[0]) + self.skip(x[1])

class snnBlock(nn.Module):
    def __init__(self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        vth:int=1,
        leak:int=1) -> None:
        super(snnBlock, self).__init__()
        # stride should be 1, x, x? x, x, 1
        conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, 
            kernel_size=3, stride=1, padding=1, bias=False)
        self.neuron1 = Neuron(conv1, vth, leak)
        conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, 
            kernel_size=3, stride=stride, padding=1, bias=False)
        conv3 = nn.Conv2d(in_channels=inplanes, out_channels=planes, 
            kernel_size=1, stride=stride, padding=0, bias=False)
        self.neuron2 = Neuron(ResponseFunc_skip(conv2, conv3), vth, leak)

    def forward(self, x):
        m = self.neuron1(x)
        return self.neuron2((m, x))
    
    def snn_forward(self, x):
        m = self.neuron1.snn_forward(x)
        return self.neuron2.snn_forward((m, x))

class snnResnet(nn.Module):
    def __init__(self,
        layers: List[int],
        in_channel = 2,
        num_classes:int=10,
        HW:tuple=(42, 42),

        vth:int=1,
        leak:int=1) -> None:

        super(snnResnet, self).__init__()

        conv11 = nn.Conv2d(in_channels=in_channel, out_channels=64, 
            kernel_size=3, stride=1, padding=1, bias=False)
        self.neuron1 = Neuron(conv11, vth, leak)
        avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.neuronp1 = Neuron(avgpool1, 0.75, leak, pool=True)
        self.layer1 = self._make_layers(layers[0], 64, 128, stride=1, vth=vth, leak=leak)
        self.layer2 = self._make_layers(layers[1], 128, 256, stride=2, vth=vth, leak=leak)
        self.layer3 = self._make_layers(layers[2], 256, 512, stride=2, vth=vth, leak=leak)
        H, W = int(HW[0]/2), int(HW[1]/2)
        for _ in range(layers[1]+layers[2]):
            H, W = int((H+1)/2), int((W+1)/2)
        fc1 = nn.Linear(512 * H*W, 1024, bias=False)
        self.neuronf1 = Neuron(fc1, vth, leak)
        self.fc2 = nn.Linear(1024, num_classes, bias=False)
        self.vth = vth

    def snn_forward(self, x):
        x = self.neuron1.snn_forward(x)
        x = self.neuronp1.snn_forward(x)
        for m in self.layer1:
            x = m.snn_forward(x)
        for m in self.layer2:
            x = m.snn_forward(x)
        for m in self.layer3:
            x = m.snn_forward(x)
        x = torch.flatten(x, 1)
        x = self.neuronf1.snn_forward(x)
        return x
    
    def forward(self, x):
        x = self.neuron1(x)
        x = self.neuronp1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.flatten(x, 1)
        x = self.neuronf1(x)
        x = self.fc2(x)
        return x / self.vth
    
    def updateScale(self, sign):
        for m in self.modules():
            if isinstance(m, Neuron):
                m.updateScale(sign)

    def reset_mem(self):
        for m in self.modules():
            if isinstance(m, Neuron):
                m.reset(0, 0, 0)

    def _make_layers(self, num, in_channel, out_channel, stride, vth, leak):
        layers = []
        layers.append(snnBlock(in_channel, out_channel, stride, vth, leak))
        for _ in range(1, num):
            layers.append(snnBlock(out_channel, out_channel, stride, vth, leak))
        return nn.Sequential(*layers)

class snnResnet34(nn.Module):
    def __init__(self,
        layers: List[int],
        in_channel = 2,
        num_classes:int=10,
        HW:tuple=(42, 42),

        vth:int=1,
        leak:int=1) -> None:

        super(snnResnet34, self).__init__()

        conv11 = nn.Conv2d(in_channels=in_channel, out_channels=64, 
            kernel_size=3, stride=1, padding=1, bias=False)
        self.neuron1 = Neuron(conv11, vth, leak)
        avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.neuronp1 = Neuron(avgpool1, 0.75, leak, pool=True)
        self.layer1 = self._make_layers(layers[0], 64, 128, stride=1, vth=vth, leak=leak)
        self.layer2 = self._make_layers(layers[1], 128, 256, stride=2, vth=vth, leak=leak)
        self.layer3 = self._make_layers(layers[2], 256, 512, stride=2, vth=vth, leak=leak)
        self.layer4 = self._make_layers(layers[3], 512, 512, stride=2, vth=vth, leak=leak)
        H, W = int(HW[0]/2), int(HW[1]/2)
        for _ in range(layers[1]+layers[2]+layers[3]):
            H, W = int((H+1)/2), int((W+1)/2)
        fc1 = nn.Linear(512 * H*W, 1024, bias=False)
        self.neuronf1 = Neuron(fc1, vth, leak)
        self.fc2 = nn.Linear(1024, num_classes, bias=False)
        self.vth = vth

    def snn_forward(self, x):
        x = self.neuron1.snn_forward(x)
        x = self.neuronp1.snn_forward(x)
        for m in self.layer1:
            x = m.snn_forward(x)
        for m in self.layer2:
            x = m.snn_forward(x)
        for m in self.layer3:
            x = m.snn_forward(x)
        for m in self.layer4:
            x = m.snn_forward(x)
        x = torch.flatten(x, 1)
        x = self.neuronf1.snn_forward(x)
        return x
    
    def forward(self, x):
        x = self.neuron1(x)
        x = self.neuronp1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)
        x = self.neuronf1(x)
        x = self.fc2(x)
        return x / self.vth
    
    def updateScale(self, sign):
        for m in self.modules():
            if isinstance(m, Neuron):
                m.updateScale(sign)

    def reset_mem(self):
        for m in self.modules():
            if isinstance(m, Neuron):
                m.reset(0, 0, 0)

    def _make_layers(self, num, in_channel, out_channel, stride, vth, leak):
        layers = []
        layers.append(snnBlock(in_channel, out_channel, stride, vth, leak))
        for _ in range(1, num):
            layers.append(snnBlock(out_channel, out_channel, stride, vth, leak))
        return nn.Sequential(*layers)

def snnresnet9(vth, leak, num_classes, CHW)->snnResnet:
    return snnResnet([1, 1, 1], CHW[0], num_classes, (CHW[1], CHW[2]),vth, leak)

def snnresnet18(vth, leak, num_classes, CHW)->snnResnet:
    return snnResnet([3, 2, 2], CHW[0], num_classes, (CHW[1], CHW[2]),vth, leak)

def snnresnet34(vth, leak, num_classes, CHW)->snnResnet:
    return snnResnet34([3, 4, 6, 3], CHW[0], num_classes, (CHW[1], CHW[2]),vth, leak)
