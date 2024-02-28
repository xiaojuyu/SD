from .node import Neuron

from tkinter.tix import IMMEDIATE
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os.path import join, dirname, isfile
# SNN
class ALTP_D(nn.Module):
    def __init__(self, shape, ones=True, bias: bool = False, device=None) -> None:
        factory_kwargs = {'device': device}
        super(ALTP_D, self).__init__()
        self.shape = shape
        self.ones = ones
        self.weight = torch.nn.parameter.Parameter(torch.empty(shape, **factory_kwargs))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.empty(shape, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.ones:
            torch.nn.init.normal_(self.weight, mean=0.2339, std=0.3971) # from s150's ckp
        else:
            torch.nn.init.zeros_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor):
        re = input.mul(self.weight.relu())
        if self.bias is not None:
            re = re.add(self.bias)
        return re

    def extra_repr(self) -> str:
        return 'shape={}, bias={}'.format(self.shape, self.bias is not None)

class ResponseFunc(nn.Module):
    def __init__(self, separate, share_func) -> None:
        super(ResponseFunc, self).__init__()
        self.separate = separate
        self.share_func = share_func

    def forward(self, x):
        out = self.separate(x)
        return self.share_func(out.detach())+out

class ALTP_block(Neuron):
    def __init__(self,
        block_type,
        conv_or_fc_in_channels=2,
        conv_or_fc_out_channels=64,
        conv_or_pool_kernel_size:int=3,
        conv_or_pool_stride:int=1,
        conv_or_pool_padding:int=1,
        conv_or_fc_in_bias:bool=False,
        add_ALTP:bool=False,
        ALTP_shape=(),
        ALTP_ones=None,
        ALTP_bias: bool = False,
        ALTP_device=None,
        Neuron_threshold=1,
        Neuron_leak=1,
        Neuron_scale:int=1,
        Neuron_pool:bool=False,
        Neuron_out:bool=False,
        Neuron_name:str="neuron",
        ) -> None:
        if block_type == 'conv':
            response_func = nn.Conv2d(conv_or_fc_in_channels, conv_or_fc_out_channels, conv_or_pool_kernel_size, conv_or_pool_stride, conv_or_pool_padding, conv_or_fc_in_bias)
        elif block_type == 'maxpool':
            response_func = nn.AvgPool2d(conv_or_pool_kernel_size)
        elif block_type == 'avgpool':
            response_func = nn.AvgPool2d(conv_or_pool_kernel_size)
        elif block_type == 'fc':
            response_func = nn.Linear(conv_or_fc_in_channels, conv_or_fc_out_channels, conv_or_fc_in_bias)
        else:
            raise ValueError("Unable to define %s" % (block_type))
        if add_ALTP:
            share_response_func = ALTP_D(ALTP_shape, ALTP_ones, ALTP_bias, ALTP_device)
            response_func = ResponseFunc(response_func, share_response_func)
        super().__init__(response_func, Neuron_threshold, Neuron_leak, Neuron_scale, Neuron_pool, Neuron_out, Neuron_name)
        
# EVENT Represent
class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
        if isfile(path):
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None,...,None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in range(1000):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()


    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels-1)] = 0
        gt_values[ts > 1.0 / (num_channels-1)] = 0

        return gt_values

class QuantizationLayerEST(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1)):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=dim[0])
        self.dim = dim

    def forward(self, events):
        # points is a list, since events can have any size
        # B = 1 #int((1+events[-1,-1]).item())
        B = int(1+events[-1,-1].item())
        if B < 1:
            B = 1
        num_voxels = int(2 * np.prod(self.dim) * B)
        vox = events[0].new_full([num_voxels,], fill_value=0)
        C, H, W = self.dim

        # get values for each channel
        x, y, t, p, b= events.t()

        # normalizing timestamps
        t = t / t.max()


        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b \


        for i_bin in range(C):
           # values = t * self.value_layer.forward(t-i_bin/(C-1))
            values = t * self.value_layer.trilinear_kernel(t-i_bin/(C-1), C)

            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin
            vox.put_(idx.long(), values, accumulate=True)

        # vox[vox > 0] = 1
        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1) # 第二维对应的两个四维拼接到一起

        return vox

class QuantizationLayerVoxGrid(nn.Module):
    def __init__(self, dim):
        nn.Module.__init__(self)
        self.dim = dim

    def forward(self, events):
        epsilon = 10e-3
        # B = 1# int(1+events[-1,-1].item())
        B = int(1+events[-1,-1].item())
        if B < 1:
            B = 1
        num_voxels = int(np.prod(self.dim) * B)
        vox_grid = events[0].new_full([num_voxels,], fill_value=0)
        C, H, W = self.dim

        # get values for each channel
        x, y, t, p, b = events.t()

        # normalizing timestamps
        t = t / t.max()

        for i_bin in range(C):
            index = (t > i_bin/C) & (t <= (i_bin+1)/C)
            x1 = x[index]
            y1 = y[index]
            b1 = b[index]
            
            idx = x1 + W*y1 + W*H*i_bin + C*H*W*b1
            val = torch.zeros_like(x1) + 1
            vox_grid.put_(idx.long(), val, accumulate=True)

        # normalize
     #   vox_grid = vox_grid / (vox_grid.max() + epsilon)
        vox_grid[vox_grid > 0] = 1
        vox_grid = vox_grid.view(-1, C, H, W)
        return vox_grid

class QuantizationLayerEventCount(nn.Module):
    def __init__(self, dim):
        nn.Module.__init__(self)
        self.dim = dim

    def forward(self, events):
        epsilon = 10e-3 # avoid divide by zero
        # B = 1 # int(1+events[-1,-1].item())
        B = int(1+events[-1,-1].item())
        if B < 1:
            B = 1
        num_voxels = int(2 * np.prod(self.dim) * B) # 照片大小：np.prod(self.dim) 极性：2 batch_size:B
        vox_ec = events[0].new_full([num_voxels,], fill_value=0) # event counts

        H, W = self.dim 

        # get values for each channel
        x, y, t, p, b= events.t()

        # normalizing timestamps
        t = t / t.max()

        idx = x + W*y + W*H*p + W*H*2*b
        val = torch.zeros_like(x) + 1
        vox_ec.put_(idx.long(), val, accumulate=True)

        # normalize 
     #   vox_ec = (vox_ec-vox_ec.mean()) / (vox_ec.max() + epsilon)
        # vox_ec[vox_ec > 0] = 1
        vox_ec = vox_ec.view(-1, 2, H, W)
        
        return vox_ec

class QuantizationLayerEventFrame(nn.Module):
    def __init__(self, dim):
        nn.Module.__init__(self)
        self.dim = dim
        self.quantization_layer = QuantizationLayerEventCount(dim)

    def forward(self, events):
        vox = self.quantization_layer.forward(events)
        event_frame = vox.sum(dim=1)
        H, W = self.dim
        event_frame = event_frame.view(-1, 1, H, W)

        return event_frame

class QuantizationLayerEventFeature(nn.Module):
    def __init__(self, dim):
        nn.Module.__init__(self)
        self.dim = dim
        est_dim = (9,dim[0],dim[1])
        vg_dim = (9,dim[0],dim[1])
        ef_dim = dim
        ec_dim = dim
        self.est = QuantizationLayerEST(est_dim)
        self.vg = QuantizationLayerVoxGrid(vg_dim)
        self.ef = QuantizationLayerEventFrame(ef_dim)
        self.ec = QuantizationLayerEventCount(ec_dim)

    def forward(self, events):
        event_est = self.est.forward(events)
        event_vg = self.vg.forward(events)
        event_ef = self.ef.forward(events)
        event_ec = self.ec.forward(events)
        event_feature = [event_est,event_vg,event_ef,event_ec]
        event_feature = torch.cat(event_feature, dim=1)
        return event_feature
