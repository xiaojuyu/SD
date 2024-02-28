import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
import logging
from torch.utils.tensorboard import SummaryWriter


class SelfAttention(nn.Module):
    def __init__(self, channels, H, W):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.H = H
        self.W = W
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.H * self.W).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.H, self.W)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, c_base=64, c_mult=[1,2,4,8], img_size=[1,1,32,32], time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, c_base * c_mult[0])
        self.down1 = Down(c_base * c_mult[0], c_base * c_mult[1])
        self.sa1 = SelfAttention(c_base * c_mult[1], img_size[2] // 2, img_size[3] // 2)
        self.down2 = Down(c_base * c_mult[1], c_base * c_mult[2])
        self.sa2 = SelfAttention(c_base * c_mult[2], img_size[2] // 4, img_size[3] // 4)
        self.down3 = Down(c_base * c_mult[2], c_base * c_mult[2])
        self.sa3 = SelfAttention(c_base * c_mult[2], img_size[2] // 8, img_size[3] // 8)

        self.bot1 = DoubleConv(c_base * c_mult[2], c_base * c_mult[3])
        self.bot2 = DoubleConv(c_base * c_mult[3], c_base * c_mult[3])
        self.bot3 = DoubleConv(c_base * c_mult[3], c_base * c_mult[2])

        self.up1 = Up(c_base * c_mult[3], c_base * c_mult[1])
        self.sa4 = SelfAttention(c_base * c_mult[1], img_size[2] // 4, img_size[3] // 4)
        self.up2 = Up(c_base * c_mult[2], c_base * c_mult[0])
        self.sa5 = SelfAttention(c_base * c_mult[0], img_size[2] // 2, img_size[3] // 2)
        self.up3 = Up(c_base * c_mult[1], c_base * c_mult[0])
        self.sa6 = SelfAttention(c_base * c_mult[0], img_size[2], img_size[3])
        self.outc = nn.Conv2d(c_base * c_mult[0], c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output

class Diffusion:
    def __init__(self, noise_type='spiking', noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=[10,3,32,32], step=None, device="cuda"):
        self.noise_type = noise_type
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.step = step

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def get_spiking_noise(self, x, p = 0.5):
        index = torch.rand_like(x.unsqueeze(4).repeat(1,1,1,1,self.step)) < p
        noise = torch.ones_like(x.unsqueeze(4).repeat(1,1,1,1,self.step))
        noise[index] = -1
        return noise.sum(4)
    
    def get_gauss_noise(self, x):
        noise = torch.randn_like(x)
        return noise
    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        if self.noise_type == 'spiking':
            noise = self.get_spiking_noise(x)
        else:
            noise = self.get_gauss_noise(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_ddpm(self, model, device, n=None):
        print("Sampling images....")
        model.eval()
        with torch.no_grad():
            # 随机采样
            if n is None:
                x = torch.randn(self.img_size).to(device)
            else:
                img_size = self.img_size
                img_size[0] = n
                x = torch.randn(img_size).to(device)
            if self.noise_type == 'spiking':
                x = self.get_spiking_noise(x)
            else:
                x = self.get_gauss_noise(x)
            # 去噪
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(x.shape[0]) * i).long().to(device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    if self.noise_type == 'spiking':
                        noise = self.get_spiking_noise(x)
                    else:
                        noise = self.get_gauss_noise(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    
    def sample_ddim(self, model, device, skip, eta_flag='ddim', n=None):
        print("Sampling images....")
        seq = range(0, self.noise_steps, skip)
        seq_next = [-1] + list(seq[:-1])
        model.eval()
        with torch.no_grad():
            # 随机采样
            if n is None:
                x = torch.randn(self.img_size).to(device)
            else:
                img_size = self.img_size
                img_size[0] = n
                x = torch.randn(img_size).to(device)
            if self.noise_type == 'spiking':
                x = self.get_spiking_noise(x)
            else:
                x = self.get_gauss_noise(x)
            # 去噪
            for (i, j) in tqdm(zip(reversed(seq), reversed(seq_next))):
                t = (torch.ones(x.shape[0]) * i).long().to(device)
                next_t = (torch.ones(x.shape[0]) * j).long().to(device)
                at = self.compute_alpha(self.beta, t.long())
                at_next = self.compute_alpha(self.beta, next_t.long())
                noise = model(x, t)
                x0_t = (x - noise * (1 - at).sqrt()) / at.sqrt()
                if eta_flag == 'ddim':
                    eta = 0
                else:
                    eta = (1 - at_next) / (1 - at).sqrt() * (1 - at / at_next).sqrt()
                c1 = (
                    eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                if self.noise_type == 'spiking':
                    x = at_next.sqrt() * x0_t + c1 * self.get_spiking_noise(x) + c2 * noise
                else:
                    x = at_next.sqrt() * x0_t + c1 * self.get_gauss_noise(x) + c2 * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    
    def sample_test(self, model, x, device):
        images_forward_x = []
        images_forward_Ɛ = []
        images_backward_x = []
        images_backward_Ɛ = []

        with torch.no_grad():
            # 加噪
            forward_x = x
            for i in range(1, self.noise_steps):
                t = (torch.ones(x.shape[0]) * i).long().to(device)
                forward_x, forward_Ɛ = self.noise_images(x, t)
                images_forward_x.append(forward_x)
                images_forward_Ɛ.append(forward_Ɛ)
                print('t:{} forward_x:{} forward_Ɛ:{}'.format(i, forward_x.sum(), forward_Ɛ.sum()))

            # 去噪
            backward_x = forward_x
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(x.shape[0]) * i).long().to(device)
                backward_Ɛ = model(backward_x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    if self.noise_type == 'spiking':
                        noise = self.get_spiking_noise(x)
                    else:
                        noise = self.get_gauss_noise(x)
                else:
                    noise = torch.zeros_like(backward_x).to(device)
                backward_x = 1 / torch.sqrt(alpha) * (backward_x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * backward_Ɛ) + torch.sqrt(beta) * noise
                images_backward_x.append(backward_x)
                images_backward_Ɛ.append(backward_Ɛ)
                print('t:{} backward_x:{} backward_Ɛ:{}'.format(i, backward_x.sum(), backward_Ɛ.sum()))
        return images_forward_x, images_forward_Ɛ, images_backward_x, images_backward_Ɛ