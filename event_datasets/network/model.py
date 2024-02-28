from .layer import QuantizationLayerEST, QuantizationLayerEventCount, QuantizationLayerEventFrame, QuantizationLayerVoxGrid, QuantizationLayerEventFeature
from .layer import ALTP_block, ALTP_D
from .node import Neuron
from .vgg import vgg19, vgg7, snnvgg7, snnvgg19
from .resnet import resnet9, resnet18, resnet34, resnet50, resnet101, snnresnet9, snnresnet18, snnresnet34
from torchvision.models.mobilenet import mobilenet_v2

import math
import torch
import torch.nn as nn

def Classifier(
    representation:str, 
    model_name:str, 
    img_dim:tuple, 
    in_channel:int,
    num_class:int, 
    repeat:int=1, 
    avgtimes:int=1, 
    vth:int=1, 
    leak:int=1, 
    pretrained:bool=True) -> nn.Module:
    '''
    build a model with arch 'model_name', the data will be handled by representation method.
    img_dim is not include channel(default 2 for event data).
    Specifically, for eventcount and eventframe, the img_dim is (H, W)
    for est and voxgrid, the img_dim is (voxel, H, W)
    for timeSteps, the img_dim is (H, W, T)
    The repeat and avgtimes is used for reproduce the AAAI2021 paper, 
    which can be found at https://www.aaai.org/AAAI21Papers/AAAI-4138.WuHao.pdf
    we adapt this paper as SNN's baseline.
    '''
    representation = representation.lower()
    model_name = model_name.lower()
    if representation == 'timesteps':
        return SNNClassifier(img_dim, num_class, model_name, avgtimes, repeat, vth, leak)
    elif representation in ['est', 'voxgrid', 'eventcount', 'eventframe', 'eventfeature']:
        return CNNClassifier(img_dim, num_class, in_channel, representation, model_name, pretrained)
    elif representation in ['origin', 'origin_cifar10', 'origin_caltech101']:
        return CNNClassifierOrigin(num_class, in_channel, model_name, pretrained)

  
class CNNClassifier(nn.Module):
    def __init__(self,
                 voxel_dimension=(9,128,128),  # dimension of voxel will be C x 2 x H x W
                 num_classes=10,
                 in_channel=2,
                 representation='est', #'EventCount', 'VoxGrid', 'EventFrame'
                 model_name = 'vgg19',
                 pretrained=True):

        nn.Module.__init__(self)
        
        if representation == 'eventcount':
            self.quantization_layer = QuantizationLayerEventCount(voxel_dimension)
        elif representation == 'eventframe':
            self.quantization_layer = QuantizationLayerEventFrame(voxel_dimension)
        elif representation == 'voxgrid':
            self.quantization_layer = QuantizationLayerVoxGrid(voxel_dimension)
        elif representation == 'est':
            self.quantization_layer = QuantizationLayerEST(voxel_dimension)
        elif representation == 'eventfeature':
            self.quantization_layer = QuantizationLayerEventFeature(voxel_dimension)
        elif representation == 'origin':
            self.quantization_layer = QuantizationLayerEventFeature(voxel_dimension)

        # replace fc layer and first convolutional layer
        input_channels = in_channel # for event count and event frame
        if len(voxel_dimension) == 3: # for vox grid and EST
            input_channels *= voxel_dimension[0]

        if model_name == 'vgg7':
            self.classifier = vgg7(pretrained=pretrained)
            self.classifier.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
            self.classifier.classifier[-1] = nn.Linear(self.classifier.classifier[-1].in_features, num_classes)
            # self.classifier.classifier[0] = nn.Linear(256*int(voxel_dimension[-1]/4)*int(voxel_dimension[-2]/4), 1024, bias=False)
        elif model_name == 'resnet9':
            self.classifier = resnet9(pretrained=pretrained)
            self.classifier.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.classifier.classifier.fc2 = nn.Linear(self.classifier.classifier.fc2.in_features, num_classes)
            # self.classifier.classifier.fc2 = nn.Linear(512*int((voxel_dimension[-1]-2)/8)*int((voxel_dimension[-2]-2)/8), num_classes, bias=False)
        elif model_name == 'vgg19':
            self.classifier = vgg19(pretrained=pretrained)   # nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            self.classifier.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
            self.classifier.classifier[-1] = nn.Linear(4096, num_classes, bias=True)
        elif model_name == 'resnet18':
            self.classifier = resnet18(pretrained=pretrained)
            self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
        elif model_name == 'resnet34':
            self.classifier = resnet34(pretrained=pretrained)
            self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
        elif model_name == 'resnet50':
            self.classifier = resnet50(pretrained=pretrained)
            self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
        elif model_name == 'resnet101':
            self.classifier = resnet101(pretrained=pretrained)
            self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
        elif model_name == 'mobilenet_v2':
            self.classifier = mobilenet_v2(pretrained=pretrained)
            self.classifier.features[0][0] = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.classifier.classifier[-1] = nn.Linear(1280, num_classes, bias=True)

    def forward(self, x):
        vox = self.quantization_layer.forward(x).float()
        pred = self.classifier.forward(vox)
        return pred


class SNNClassifier(nn.Module):
    def __init__(self,
        img_dim,
        num_classes,
        model_name:str,
        avgtimes:int=1,
        repeat:int=1,
        vth:int=1,
        leak:int=1
        ) -> None:

        super(SNNClassifier, self).__init__()
        
        self.steps = img_dim[-1]
        self.repeat = repeat
        self.avgtimes = avgtimes
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=3)
        if model_name.lower() == 'vgg7':
            self.arch = snnvgg7(vth, leak, num_classes, (2, int(img_dim[0]/3), int(img_dim[1]/3)))           
        elif model_name.lower() == 'resnet9':
            self.arch = snnresnet9(vth, leak, num_classes, (2, int(img_dim[0]/3), int(img_dim[1]/3)))    
        elif model_name.lower() == 'vgg19':
            self.arch = snnvgg19(vth, leak, num_classes, (2, int(img_dim[0]/3), int(img_dim[1]/3)))           
        elif model_name.lower() == 'resnet18':
            self.arch = snnresnet18(vth, leak, num_classes, (2, int(img_dim[0]/3), int(img_dim[1]/3)))      
        elif model_name.lower() == 'resnet34':
            self.arch = snnresnet34(vth, leak, num_classes, (2, int(img_dim[0]/3), int(img_dim[1]/3)))     
        
        self._weight_init()
    
    def forward(self, x, updateSign=None):
        self.arch.reset_mem()
        total_input = 0
        with torch.no_grad():
            for i in range(self.steps * self.repeat):
                # Poisson input spike generation
                eventframe_input = x[:, :, :, :, i % self.steps].float()
                eventframe_input = self.avgtimes * self.downsample(eventframe_input)
                total_input += eventframe_input
                self.arch.snn_forward(eventframe_input)

            self.arch.updateScale(updateSign)
        
        out = self.arch(total_input)
        return out / self.steps

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(2.0 / n)
                m.weight.data.normal_(0, variance1)

            elif isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_in = size[1]  # number of columns
                variance2 = math.sqrt(2.0 / fan_in)
                m.weight.data.normal_(0.0, variance2)


class CNNClassifierOrigin(nn.Module):
    def __init__(self,
                 num_classes=10,
                 in_channel=2,
                 model_name = 'vgg19',
                 pretrained=True):

        nn.Module.__init__(self)

        # replace fc layer and first convolutional layer
        input_channels = in_channel # for event count and event frame

        if model_name == 'vgg7':
            self.classifier = vgg7(pretrained=pretrained)
            self.classifier.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
            self.classifier.classifier[-1] = nn.Linear(self.classifier.classifier[-1].in_features, num_classes)
            # self.classifier.classifier[0] = nn.Linear(256*int(voxel_dimension[-1]/4)*int(voxel_dimension[-2]/4), 1024, bias=False)
        elif model_name == 'resnet9':
            self.classifier = resnet9(pretrained=pretrained)
            self.classifier.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.classifier.classifier.fc2 = nn.Linear(self.classifier.classifier.fc2.in_features, num_classes)
            # self.classifier.classifier.fc2 = nn.Linear(512*int((voxel_dimension[-1]-2)/8)*int((voxel_dimension[-2]-2)/8), num_classes, bias=False)
        elif model_name == 'vgg19':
            self.classifier = vgg19(pretrained=pretrained)   # nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            self.classifier.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
            self.classifier.classifier[-1] = nn.Linear(4096, num_classes, bias=True)
        elif model_name == 'resnet18':
            self.classifier = resnet18(pretrained=pretrained)
            self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
        elif model_name == 'resnet34':
            self.classifier = resnet34(pretrained=pretrained)
            self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
        elif model_name == 'mobilenet_v2':
            self.classifier = mobilenet_v2(pretrained=pretrained)
            self.classifier.features[0][0] = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.classifier.classifier[-1] = nn.Linear(1280, num_classes, bias=True)

    def forward(self, x):
        pred = self.classifier.forward(x)
        return pred
# class SNN(nn.Module):
#     def __init__(self, size, repeat, avgtimes, sign, device, vth, leak) -> None:
#         super(SNN, self).__init__()
#         self.repeat = repeat
#         self.avgtimes = avgtimes
#         self.sign = sign
#         self.vth = vth
#         self.leak = leak
#         H,W = size # list:[H,W]
#         para_size_H = H
#         para_size_W = W
#         device = torch.device(f'cuda:{device}')
#         self.maxpool4 = nn.AvgPool2d(kernel_size=3)
#         para_size_H = int(para_size_H/3)
#         para_size_W = int(para_size_W/3)
#         conv11 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.neuron11 = Neuron(conv11, self.vth, self.leak)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.neuron12 = Neuron(conv12, self.vth, self.leak)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         avgpool1 = nn.AvgPool2d(kernel_size=2)
#         self.neuronp1 = Neuron(avgpool1, 0.75, self.leak, pool=True)
#         para_size_H = int(para_size_H/2)
#         para_size_W = int(para_size_W/2)
#         conv21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.neuron21 = Neuron(conv21, self.vth, self.leak)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         conv22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.neuron22 = Neuron(conv22, self.vth, self.leak)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         conv23 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
#         self.neuron23 = Neuron(conv23, self.vth, self.leak)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         self.avgpool2 = nn.MaxPool2d(kernel_size=2)
#         para_size_H = int(para_size_H/2)
#         para_size_W = int(para_size_W/2)

#         fc0 = nn.Linear(256 * para_size_H * para_size_W, 1024, bias=False)
#         self.neuronf0 = Neuron(fc0, self.vth, self.leak)
#         self.classifier = nn.Linear(1024, 10, bias=False)
#         self.classifier.threshold = self.vth
        
#         self._weight_init()

#     def forward(self, input_x):
#         self.reset_mem()
#         total_input = 0
#         with torch.no_grad():
#             for i in range(self.steps * self.repeat):
#                 # Poisson input spike generation
#                 eventframe_input = input_x[:, :, :, :, i % self.steps].float()
#                 eventframe_input = self.avgtimes * self.maxpool4(eventframe_input)
#                 total_input += eventframe_input
#                 spike = self.neuron11.snn_forward(eventframe_input)
#                 spike = self.neuron12.snn_forward(spike)
#                 spike = self.neuronp1.snn_forward(spike)
#                 spike = self.neuron21.snn_forward(spike)
#                 spike = self.neuron22.snn_forward(spike)
#                 spike = self.neuron23.snn_forward(spike)

#                 fc_in = self.avgpool2(spike).view(input_x.size(0), -1)

#                 spike = self.neuronf0.snn_forward(fc_in)
#             self.updateScale(self.sign)
#         with torch.enable_grad():
#             spike = self.neuron11(total_input)
#             spike = self.neuron12(spike)
#             spike = self.neuronp1(spike)
#             spike = self.neuron21(spike)
#             spike = self.neuron22(spike)
#             spike = self.neuron23(spike)

#             fc_in = self.avgpool2(spike).view(input_x.size(0), -1)

#             character = self.neuronf0(fc_in)
#             out = self.classifier(character)
#         return out / self.classifier.threshold / self.steps
    

        

#     def updateScale(self, sign):
#         for m in self.modules():
#             if isinstance(m, Neuron):
#                 m.updateScale(sign)

#     def reset_mem(self):
#         for m in self.modules():
#             if isinstance(m, Neuron):
#                 m.reset(0, 0, 0)
        
#     def _weight_init(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
#                 variance1 = math.sqrt(2.0 / n)
#                 m.weight.data.normal_(0, variance1)

#             elif isinstance(m, nn.Linear):
#                 size = m.weight.size()
#                 fan_in = size[1]  # number of columns
#                 variance2 = math.sqrt(2.0 / fan_in)
#                 m.weight.data.normal_(0.0, variance2)


# class SNN1(nn.Module):
#     def __init__(self, args) -> None:
#         self.args = args
#         super(SNN1, self).__init__()
#         device = torch.device(f'cuda:{self.args.device}')
#         # maxpool
#         self.maxpool1 = nn.AvgPool2d(kernel_size=3)
#         # conv1
#         self.conv1 = ALTP_block(block_type='conv',conv_or_fc_in_channels=2,conv_or_fc_out_channels=64,conv_or_pool_kernel_size=3,
#         conv_or_pool_stride=1,conv_or_pool_padding=1,conv_or_fc_in_bias=False,Neuron_threshold=self.self.vth,Neuron_leak=self.self.leak)
#         # conv2
#         self.conv2 = ALTP_block(block_type='conv',conv_or_fc_in_channels=64,conv_or_fc_out_channels=64,conv_or_pool_kernel_size=3,
#         conv_or_pool_stride=1,conv_or_pool_padding=1,conv_or_fc_in_bias=False,Neuron_threshold=self.self.vth,Neuron_leak=self.self.leak)
#         # avgpool
#         self.avgpool1 = ALTP_block(block_type='avgpool',conv_or_pool_kernel_size=2,Neuron_threshold=0.75,Neuron_leak=self.self.leak,Neuron_pool=True)
#         # conv3
#         self.conv3 = ALTP_block(block_type='conv',conv_or_fc_in_channels=64,conv_or_fc_out_channels=128,conv_or_pool_kernel_size=3,
#         conv_or_pool_stride=1,conv_or_pool_padding=1,conv_or_fc_in_bias=False,Neuron_threshold=self.self.vth,Neuron_leak=self.self.leak)
#         # conv4
#         self.conv4 = ALTP_block(block_type='conv',conv_or_fc_in_channels=128,conv_or_fc_out_channels=128,conv_or_pool_kernel_size=3,
#         conv_or_pool_stride=1,conv_or_pool_padding=1,conv_or_fc_in_bias=False,Neuron_threshold=self.self.vth,Neuron_leak=self.self.leak)
#         # conv5
#         self.conv5 = ALTP_block(block_type='conv',conv_or_fc_in_channels=128,conv_or_fc_out_channels=256,conv_or_pool_kernel_size=3,
#         conv_or_pool_stride=1,conv_or_pool_padding=1,conv_or_fc_in_bias=False,Neuron_threshold=self.self.vth,Neuron_leak=self.self.leak)
#         # avgpool
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2)
#         # fc1000
#         self.fc1000 = ALTP_block(block_type='fc',conv_or_fc_in_channels=256*10*10,conv_or_fc_out_channels=1024,add_ALTP=True,ALTP_shape=(1024),
#         ALTP_device=device,Neuron_threshold=self.self.vth,Neuron_leak=self.self.leak)
#         # fc10
#         self.classifier = nn.Linear(1024, 10, bias=False)
#         self.classifier.threshold = self.self.vth
        
#         self._weight_init()

#     def forward(self, input_x, steps, l=1):
#         self.reset_mem()
#         total_input = 0
#         with torch.no_grad():
#             for i in range(steps * self.self.repeat):
#                 # Poisson input spike generation
#                 eventframe_input = input_x[:, :, :, :, i % steps].float()
#                 eventframe_input = self.self.avgtimes * self.maxpool1(eventframe_input)
#                 total_input += eventframe_input
#                 spike = self.conv1.snn_forward(eventframe_input)
#                 spike = self.conv2.snn_forward(spike)
#                 spike = self.avgpool1.snn_forward(spike)
#                 spike = self.conv3.snn_forward(spike)
#                 spike = self.conv4.snn_forward(spike)
#                 spike = self.conv5.snn_forward(spike)
#                 fc_in = self.maxpool2(spike).view(input_x.size(0), -1)
#                 spike = self.fc1000.snn_forward(fc_in)
#             self.updateScale(self.self.sign)
#         with torch.enable_grad():
#             spike = self.conv1(total_input)
#             spike = self.conv2(spike)
#             spike = self.avgpool1(spike)
#             spike = self.conv3(spike)
#             spike = self.conv4(spike)
#             spike = self.conv5(spike)

#             fc_in = self.maxpool2(spike).view(input_x.size(0), -1)

#             character = self.fc1000(fc_in)
#             out = self.classifier(character)
#         return out / self.classifier.threshold / steps
    
#     def tst(self, input_x, steps=100, l=1):
#         # 注意这里没有reset 
#         # 效果跑完再和上面合并，并且out可以测试分开conv和集中conv的效果，上面训练也可以改成集中conv
#         self.reset_mem()
#         out = 0
#         with torch.no_grad():
#             for i in range(steps * self.self.repeat):
#                 # Poisson input spike generation
#                 eventframe_input = (input_x[:, :, :, :, (i % steps)]).float()
#                 eventframe_input = self.self.avgtimes * self.maxpool1(eventframe_input)
#                 spike = self.conv1.snn_forward(eventframe_input)
#                 spike = self.conv2.snn_forward(spike)
#                 spike = self.avgpool1.snn_forward(spike)
#                 spike = self.conv3.snn_forward(spike)
#                 spike = self.conv4.snn_forward(spike)
#                 spike = self.conv5.snn_forward(spike)

#                 fc_in = self.maxpool2(spike).view(input_x.size(0), -1)

#                 spike = self.fc1000.snn_forward(fc_in)
#                 out += self.classifier(spike)
#             return out / self.classifier.threshold / steps
            
#     def updateScale(self, sign):
#         for m in self.modules():
#             if isinstance(m, Neuron):
#                 m.updateScale(sign)

#     def reset_mem(self):
#         for m in self.modules():
#             if isinstance(m, Neuron):
#                 m.reset(0, 0, 0)
        

#     def _weight_init(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
#                 variance1 = math.sqrt(2.0 / n)
#                 m.weight.data.normal_(0, variance1)
#                 # define threshold
#                 # m.threshold = self.vth

#             elif isinstance(m, nn.Linear):
#                 size = m.weight.size()
#                 fan_in = size[1]  # number of columns
#                 variance2 = math.sqrt(2.0 / fan_in)
#                 m.weight.data.normal_(0.0, variance2)


# class SNN_noaltp(nn.Module):
#     def __init__(self, size, repeat, avgtimes, sign, device, vth, leak) -> None:
#         super(SNN_noaltp, self).__init__()
#         self.repeat = repeat
#         self.avgtimes = avgtimes
#         self.sign = sign
#         self.vth = vth
#         self.leak = leak
#         H,W = size # list:[H,W]
#         para_size_H = H
#         para_size_W = W
#         # kernel_size = 3
#         # stride = 1
#         # padding = 1
#         device = torch.device(f'cuda:{device}')
#         self.maxpool4 = nn.AvgPool2d(kernel_size=3)
#         para_size_H = int(para_size_H/3)
#         para_size_W = int(para_size_W/3)
#         conv11 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.neuron11 = Neuron(conv11, self.vth, self.leak)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.neuron12 = Neuron(conv12, self.vth, self.leak)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         avgpool1 = nn.AvgPool2d(kernel_size=2)
#         self.neuronp1 = Neuron(avgpool1, 0.75, self.leak, pool=True)
#         para_size_H = int(para_size_H/2)
#         para_size_W = int(para_size_W/2)
#         conv21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.neuron21 = Neuron(conv21, self.vth, self.leak)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         conv22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.neuron22 = Neuron(conv22, self.vth, self.leak)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         conv23 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
#         self.neuron23 = Neuron(conv23, self.vth, self.leak)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         self.avgpool2 = nn.MaxPool2d(kernel_size=2)
#         para_size_H = int(para_size_H/2)
#         para_size_W = int(para_size_W/2)

#         fc0 = nn.Linear(256 * para_size_H * para_size_W, 1024, bias=False)
#         # shareF0 = ALTP_D((1024), device=device)
#         # self.neuronf0 = Neuron(ResponseFunc(fc0, shareF0), self.vth, self.leak)
#         self.neuronf0 = Neuron(fc0, self.vth, self.leak)
#         self.classifier = nn.Linear(1024, 10, bias=False)
#         self.classifier.threshold = self.vth
        
#         self._weight_init()

#     def forward(self, input_x, steps=100, l=1):
#         self.reset_mem()
#         total_input = 0
#         with torch.no_grad():
#             for i in range(steps * self.repeat):
#                 # Poisson input spike generation
#                 eventframe_input = input_x[:, :, :, :, i % steps].float()
#                 eventframe_input = self.avgtimes * self.maxpool4(eventframe_input)
#                 total_input += eventframe_input
#                 spike = self.neuron11.snn_forward(eventframe_input)
#                 spike = self.neuron12.snn_forward(spike)
#                 spike = self.neuronp1.snn_forward(spike)
#                 spike = self.neuron21.snn_forward(spike)
#                 spike = self.neuron22.snn_forward(spike)
#                 spike = self.neuron23.snn_forward(spike)

#                 fc_in = self.avgpool2(spike).view(input_x.size(0), -1)

#                 spike = self.neuronf0.snn_forward(fc_in)
#             self.updateScale(self.sign)
#         with torch.enable_grad():
#             spike = self.neuron11(total_input)
#             spike = self.neuron12(spike)
#             spike = self.neuronp1(spike)
#             spike = self.neuron21(spike)
#             spike = self.neuron22(spike)
#             spike = self.neuron23(spike)

#             fc_in = self.avgpool2(spike).view(input_x.size(0), -1)

#             character = self.neuronf0(fc_in)
#             out = self.classifier(character)
#         return out / self.classifier.threshold / steps
    
#     def tst(self, input_x, steps=100, l=1):
#         # 注意这里没有reset 
#         # 效果跑完再和上面合并，并且out可以测试分开conv和集中conv的效果，上面训练也可以改成集中conv
#         self.reset_mem()
#         out = 0
#         with torch.no_grad():
#             for i in range(steps * self.repeat):
#                 # Poisson input spike generation
#                 eventframe_input = (input_x[:, :, :, :, (i % steps)]).float()
#                 eventframe_input = self.avgtimes * self.maxpool4(eventframe_input)
#                 spike = self.neuron11.snn_forward(eventframe_input)
#                 spike = self.neuron12.snn_forward(spike)
#                 spike = self.neuronp1.snn_forward(spike)
#                 spike = self.neuron21.snn_forward(spike)
#                 spike = self.neuron22.snn_forward(spike)
#                 spike = self.neuron23.snn_forward(spike)

#                 fc_in = self.avgpool2(spike).view(input_x.size(0), -1)

#                 spike = self.neuronf0.snn_forward(fc_in)
#                 out += self.classifier(spike)
#             return out / self.classifier.threshold / steps
        

#     def updateScale(self, sign):
#         for m in self.modules():
#             if isinstance(m, Neuron):
#                 m.updateScale(sign)

#     def reset_mem(self):
#         for m in self.modules():
#             if isinstance(m, Neuron):
#                 m.reset(0, 0, 0)
        

#     def _weight_init(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
#                 variance1 = math.sqrt(2.0 / n)
#                 m.weight.data.normal_(0, variance1)
#                 # define threshold
#                 # m.threshold = self.vth

#             elif isinstance(m, nn.Linear):
#                 size = m.weight.size()
#                 fan_in = size[1]  # number of columns
#                 variance2 = math.sqrt(2.0 / fan_in)
#                 m.weight.data.normal_(0.0, variance2)


# class SNN_VGG11_noaltp(nn.Module):
#     def __init__(self, size, repeat, avgtimes, sign, device, vth, leak) -> None:
#         super(SNN_VGG11_noaltp, self).__init__()
#         self.repeat = repeat
#         self.avgtimes = avgtimes
#         self.sign = sign
#         self.vth = vth
#         self.leak = leak
#         H,W = size # list:[H,W]
#         para_size_H = H
#         para_size_W = W
#         device = torch.device(f'cuda:{device}')
#         # input tansform
#         self.maxpool4 = nn.AvgPool2d(kernel_size=3)
#         para_size_H = int(para_size_H/3)
#         para_size_W = int(para_size_W/3)
#         # block1
#         conv11 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.neuron11 = Neuron(conv11, self.vth, self.leak)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         avgpool1 = nn.AvgPool2d(kernel_size=2)
#         self.neuronp1 = Neuron(avgpool1, 0.75, self.leak, pool=True)
#         para_size_H = int(para_size_H/2)
#         para_size_W = int(para_size_W/2)
#         # block2
#         conv21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.neuron21 = Neuron(conv21, self.vth, self.leak)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         avgpool2 = nn.AvgPool2d(kernel_size=2)
#         self.neuronp1 = Neuron(avgpool2, 0.75, self.leak, pool=True)
#         para_size_H = int(para_size_H/2)
#         para_size_W = int(para_size_W/2)
#         # block3
#         conv31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
#         self.neuron31 = Neuron(conv31, self.vth, self.leak)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         conv32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
#         self.neuron32 = Neuron(conv32, self.vth, self.leak)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         avgpool3 = nn.AvgPool2d(kernel_size=2)
#         self.neuronp3 = Neuron(avgpool3, 0.75, self.leak, pool=True)
#         para_size_H = int(para_size_H/2)
#         para_size_W = int(para_size_W/2)
#         # block4
#         conv41 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
#         self.neuron41 = Neuron(conv41, self.vth, self.leak)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         conv42 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
#         self.neuron42 = Neuron(conv42, self.vth, self.leak)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         self.avgpool4 = nn.MaxPool2d(kernel_size=2)
#         para_size_H = int(para_size_H/2)
#         para_size_W = int(para_size_W/2)
#         # block5
#         conv51 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
#         self.neuron51 = Neuron(conv51, self.vth, self.leak)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         conv52 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
#         self.neuron52 = Neuron(conv52, self.vth, self.leak)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         self.avgpool5 = nn.MaxPool2d(kernel_size=2)
#         para_size_H = int(para_size_H/2)
#         para_size_W = int(para_size_W/2)
#         #fully connected
#         fc0 = nn.Linear(512 * para_size_H * para_size_W, 4096, bias=False)
#         # shareF0 = ALTP_D((1024), device=device)
#         # self.neuronf0 = Neuron(ResponseFunc(fc0, shareF0), self.vth, self.leak)
#         self.neuronf0 = Neuron(fc0, self.vth, self.leak)
#         self.dropout1 = nn.Dropout(p=0.5, inplace=False)
#         fc1 = nn.Linear(4096, 4096, bias=False)
#         self.neuronf1 = Neuron(fc1, self.vth, self.leak)
#         self.dropout1 = nn.Dropout(p=0.5, inplace=False)
#         self.classifier = nn.Linear(4096, 4096, bias=False)



#         self.classifier.threshold = self.vth
        
#         self._weight_init()

#     def forward(self, input_x, steps, l=1):
#         self.reset_mem()
#         total_input = 0
#         with torch.no_grad():
#             for i in range(steps * self.repeat):
#                 # Poisson input spike generation
#                 eventframe_input = input_x[:, :, :, :, i % steps].float()
#                 eventframe_input = self.avgtimes * self.maxpool4(eventframe_input)
#                 total_input += eventframe_input
#                 spike = self.neuron11.snn_forward(eventframe_input)
#                 spike = self.neuron12.snn_forward(spike)
#                 spike = self.neuronp1.snn_forward(spike)
#                 spike = self.neuron21.snn_forward(spike)
#                 spike = self.neuron22.snn_forward(spike)
#                 spike = self.neuron23.snn_forward(spike)

#                 fc_in = self.avgpool2(spike).view(input_x.size(0), -1)

#                 spike = self.neuronf0.snn_forward(fc_in)
#             self.updateScale(self.sign)
#         with torch.enable_grad():
#             spike = self.neuron11(total_input)
#             spike = self.neuron12(spike)
#             spike = self.neuronp1(spike)
#             spike = self.neuron21(spike)
#             spike = self.neuron22(spike)
#             spike = self.neuron23(spike)

#             fc_in = self.avgpool2(spike).view(input_x.size(0), -1)

#             character = self.neuronf0(fc_in)
#             out = self.classifier(character)
#         return out / self.classifier.threshold / steps
    
#     def tst(self, input_x, steps=100, l=1):
#         # 注意这里没有reset 
#         # 效果跑完再和上面合并，并且out可以测试分开conv和集中conv的效果，上面训练也可以改成集中conv
#         self.reset_mem()
#         out = 0
#         with torch.no_grad():
#             for i in range(steps * self.repeat):
#                 # Poisson input spike generation
#                 eventframe_input = (input_x[:, :, :, :, (i % steps)]).float()
#                 eventframe_input = self.avgtimes * self.maxpool4(eventframe_input)
#                 spike = self.neuron11.snn_forward(eventframe_input)
#                 spike = self.neuron12.snn_forward(spike)
#                 spike = self.neuronp1.snn_forward(spike)
#                 spike = self.neuron21.snn_forward(spike)
#                 spike = self.neuron22.snn_forward(spike)
#                 spike = self.neuron23.snn_forward(spike)

#                 fc_in = self.avgpool2(spike).view(input_x.size(0), -1)

#                 spike = self.neuronf0.snn_forward(fc_in)
#                 out += self.classifier(spike)
#             return out / self.classifier.threshold / steps
        

#     def updateScale(self, sign):
#         for m in self.modules():
#             if isinstance(m, Neuron):
#                 m.updateScale(sign)

#     def reset_mem(self):
#         for m in self.modules():
#             if isinstance(m, Neuron):
#                 m.reset(0, 0, 0)
        

#     def _weight_init(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
#                 variance1 = math.sqrt(2.0 / n)
#                 m.weight.data.normal_(0, variance1)
#                 # define threshold
#                 # m.threshold = self.vth

#             elif isinstance(m, nn.Linear):
#                 size = m.weight.size()
#                 fan_in = size[1]  # number of columns
#                 variance2 = math.sqrt(2.0 / fan_in)
#                 m.weight.data.normal_(0.0, variance2)


# class VGG7_bn(nn.Module):
#     def __init__(self, size, in_channel=2,num_classes=10) -> None:
#         nn.Module.__init__(self)
#         H,W = size # list:[H,W]
#         para_size_H = H
#         para_size_W = W
#         # self.maxpool_init = nn.AvgPool2d(kernel_size=3)
#         # para_size_H = int(para_size_H/3)
#         # para_size_W = int(para_size_W/3)
#         self.conv11 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn11 = nn.BatchNorm2d(64)
#         self.act11 = nn.ReLU(inplace=True)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn12 = nn.BatchNorm2d(64)
#         self.act12 = nn.ReLU(inplace=True)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2)
#         para_size_H = int(para_size_H/2)
#         para_size_W = int(para_size_W/2)
#         self.conv21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn21 = nn.BatchNorm2d(128)
#         self.act21 = nn.ReLU(inplace=True)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         self.conv22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn22 = nn.BatchNorm2d(128)
#         self.act22 = nn.ReLU(inplace=True)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         self.conv23 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn23 = nn.BatchNorm2d(256)
#         self.act23 = nn.ReLU(inplace=True)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2)
#         para_size_H = int(para_size_H/2)
#         para_size_W = int(para_size_W/2)

#         self.fc0 = nn.Linear(256 * para_size_H * para_size_W, 1024, bias=False)
#         self.actf0 = nn.ReLU(inplace=True)
#         self.fc1 = nn.Linear(1024, num_classes, bias=False)
#         self.actf1 = nn.ReLU(inplace=True)
    
#     def forward(self,input):
#         # input transform
#         # input_init = self.maxpool_init(input)
#         # block1
#         x = self.act11(self.bn11(self.conv11(input)))
#         x = self.act12(self.bn12(self.conv12(x)))
#         x = self.maxpool1(x)
#         # block2
#         x = self.act21(self.bn21(self.conv21(x)))
#         x = self.act22(self.bn22(self.conv22(x)))
#         x = self.act23(self.bn23(self.conv23(x)))
#         x = self.maxpool2(x)
#         # fully connected
#         x = x.view(x.size(0), -1)
#         x = self.actf0(self.fc0(x))
#         output = self.actf1(self.fc1(x))
#         # output
#         return output


# class VGG7(nn.Module):
#     def __init__(self, size, in_channel=2, num_classes=10) -> None:
#         nn.Module.__init__(self)
#         H,W = size # list:[H,W]
#         para_size_H = H
#         para_size_W = W
#         # self.maxpool_init = nn.AvgPool2d(kernel_size=3)
#         # para_size_H = int(para_size_H/3)
#         # para_size_W = int(para_size_W/3)
#         self.conv11 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.act11 = nn.ReLU(inplace=True)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.act12 = nn.ReLU(inplace=True)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2)
#         para_size_H = int(para_size_H/2)
#         para_size_W = int(para_size_W/2)
#         self.conv21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.act21 = nn.ReLU(inplace=True)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         self.conv22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.act22 = nn.ReLU(inplace=True)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         self.conv23 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
#         self.act23 = nn.ReLU(inplace=True)
#         para_size_H = int((para_size_H - 3 + 2 * 1 )/1) + 1
#         para_size_W = int((para_size_W - 3 + 2 * 1 )/1) + 1
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2)
#         para_size_H = int(para_size_H/2)
#         para_size_W = int(para_size_W/2)

#         self.fc0 = nn.Linear(256 * para_size_H * para_size_W, 1024, bias=False)
#         self.actf0 = nn.ReLU(inplace=True)
#         self.fc1 = nn.Linear(1024, num_classes, bias=False)
#         self.actf1 = nn.ReLU(inplace=True)
    
#     def forward(self,input):
#         # input transform
#         # input_init = self.maxpool_init(input)
#         # block1
#         x = self.act11(self.conv11(input))
#         x = self.act12(self.conv12(x))
#         x = self.maxpool1(x)
#         # block2
#         x = self.act21(self.conv21(x))
#         x = self.act22(self.conv22(x))
#         x = self.act23(self.conv23(x))
#         x = self.maxpool2(x)
#         # fully connected
#         x = x.view(x.size(0), -1)
#         x = self.actf0(self.fc0(x))
#         output = self.actf1(self.fc1(x))
#         # output
#         return output

 