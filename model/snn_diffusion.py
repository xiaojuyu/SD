import math
import torch
import einops
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from tqdm import tqdm
from spikingjelly.activation_based import functional, surrogate
from spikingjelly.activation_based.neuron import IFNode as IFNode

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim): 
        # T: total step of diff; d_model: base channel num; dim:d_model*4
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb
    
class SpikingFullySpatialTemporalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=1, attn_drop=0., proj_drop=0., T=0, H=0, W=0, tau=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads

        self.qkv_neuron = IFNode(surrogate_function=surrogate.ATan())
        self.qkv_conv = nn.Conv1d(dim, dim * 3, kernel_size=1, stride=1)
        self.qkv_bn = nn.BatchNorm1d(dim * 3)

        self.to_qkv_proj = nn.Linear(dim, dim * 3)
        self.to_qkv_bn = nn.BatchNorm1d(dim * 3)
        self.to_qkv_lif = IFNode(surrogate_function=surrogate.ATan())

        self.attn_neuron = IFNode(surrogate_function=surrogate.ATan())
        self.attn_drop = nn.Dropout(p=attn_drop)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_neuron = IFNode(surrogate_function=surrogate.ATan())
        self.proj_drop = nn.Dropout(p=proj_drop)

        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x, temb):
        # qkv
        h = self.qkv_neuron(x)
        T, B, C, H, W = h.shape
        h = h.flatten(3)
        T, B, C, N = h.shape
        h_for_qkv = h.reshape(B, C, -1).contiguous()  # B, C, T*N
        qkv_out = self.qkv_conv(h_for_qkv)
        qkv_out = self.qkv_bn(qkv_out)
        # attn
        qkv_out = self.attn_neuron(qkv_out).reshape(B, C * 3, T, N).permute(0, 2, 3, 1).contiguous().chunk(3, dim=-1)
        q, k, v = map(lambda z: einops.rearrange(z, 'b t n (h d) -> b h (t n) d', h=self.num_heads), qkv_out)
        attn = (q @ k.transpose(-2, -1))  # B, head_num, token_num, token_num
        h = (attn @ v) * 0.125
        h = h.permute(0, 2, 1, 3).reshape(B, T * N, C).reshape(B, T, N, C).permute(1, 0, 3, 2).contiguous()
        # proj
        h = self.proj_neuron(h)  # T, B, C, N
        h = h.permute(1, 2, 0, 3).contiguous()  # B, C, T, N
        h = h.reshape(B, C, -1).contiguous()  # B, C, T*N
        h = self.proj_conv(h)
        h = self.proj_bn(h).reshape(B, C, T, N).permute(2, 0, 1, 3).reshape(T, B, C, H, W).contiguous()

        return h + x


class SpikingDownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, in_ch, kernel_size=(3,3,1), stride=(2,2,1), padding=(1,1,0))
        self.bn = nn.BatchNorm3d(in_ch)
        self.neuron = IFNode(surrogate_function=surrogate.ATan())
        functional.set_step_mode(self, step_mode='m')
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x, temb):
        h = self.neuron(x)
        h = h.permute(1, 2, 3, 4, 0)  # [B, C, H, W, T]
        h = self.conv(h)
        h = self.bn(h).permute(4, 0, 1, 2, 3)  # [T, B, C, H, W]

        return h


class SpikingUpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, in_ch, kernel_size=(3,3,1), stride=(1,1,1), padding=(1,1,0))
        self.bn = nn.BatchNorm3d(in_ch)
        self.neuron = IFNode(surrogate_function=surrogate.ATan())
        functional.set_step_mode(self, step_mode='m')
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x, temb):
        T, B, C, H, W = x.shape
        h = self.neuron(x)
        h = h.flatten(0, 1)  ## [T*B C H W]
        h = F.interpolate(
            h, scale_factor=2, mode='nearest')
        _, C, H, W = h.shape
        h = h.reshape(T, B, C, H, W).contiguous() # [T, B, C, H, W]
        h = h.permute(1, 2, 3, 4, 0)  # [B, C, H, W, T]
        h = self.conv(h)
        h = self.bn(h).permute(4, 0, 1, 2, 3)  # [T, B, C, H, W]

        return h


class SpikingResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.neuron1 = IFNode(surrogate_function=surrogate.ATan())
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=(3,3,1), stride=(1,1,1), padding=(1,1,0))
        self.bn1 = nn.BatchNorm3d(out_ch)

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )

        self.neuron2 = IFNode(surrogate_function=surrogate.ATan())
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=(3,3,1), stride=(1,1,1), padding=(1,1,0))
        self.bn2 = nn.BatchNorm3d(out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
        else:
            self.shortcut = nn.Identity()

        functional.set_step_mode(self, step_mode='m')
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x, temb):
        T, B, C, H, W = x.shape

        h = self.neuron1(x)
        h = h.permute(1, 2, 3, 4, 0)  # [B, C, H, W, T]
        h = self.conv1(h)
        h = self.bn1(h).permute(4, 0, 1, 2, 3)  # [T, B, C, H, W]

        temp = self.temb_proj(temb).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, h.shape[-2], h.shape[-1])
        h = torch.add(h, temp)

        h = self.neuron2(h)
        h = h.permute(1, 2, 3, 4, 0)  # [B, C, H, W, T]
        h = self.conv2(h)
        h = self.bn2(h).permute(4, 0, 1, 2, 3)  # [T, B, C, H, W]

        h = h + self.shortcut(x.permute(1, 2, 3, 4, 0)).permute(4, 0, 1, 2, 3)

        return h

class MembraneOutputLayer(nn.Module):
    def __init__(self, timestep=4) -> None:
        super().__init__()
        self.n_steps = timestep

    def forward(self, x):
        # x: [T,N,C,H,W]
        arr = torch.arange(self.n_steps - 1, -1, -1)
        coef = torch.pow(0.8, arr).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(x.device)
        out = torch.sum(x * coef, dim=0)
        return out

class SpikingUNet(nn.Module):
    def __init__(self, noise_steps, c_in, c_out, c_base, c_mult, attn, num_res_blocks, dropout, timestep):
        super().__init__()
        tdim = c_base * 4
        self.timestep = timestep
        # Timestep embedding
        self.time_embedding = TimeEmbedding(noise_steps, c_base, tdim)
        # Begain
        self.begain_conv1 = nn.Conv3d(c_in, c_base, kernel_size=(3,3,1), stride=(1,1,1), padding=(1,1,0))
        self.begain_bn1 = nn.BatchNorm3d(c_base)
        self.begain_neuron = IFNode(surrogate_function=surrogate.ATan())
        self.begain_conv2 = nn.Conv3d(c_base, c_base, kernel_size=(3,3,1), stride=(1,1,1), padding=(1,1,0))
        self.begain_bn2 = nn.BatchNorm3d(c_base)
        # Downsampling
        self.downblocks = nn.ModuleList()
        chs = [c_base]
        now_ch = c_base
        for i, mult in enumerate(c_mult):
            out_ch = c_base * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(SpikingResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim))
                if i in attn:
                    self.downblocks.append(SpikingFullySpatialTemporalSelfAttention(out_ch))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(c_mult) - 1:
                self.downblocks.append(SpikingDownSample(now_ch))
                chs.append(now_ch)
        # Middle
        self.middleblocks = nn.ModuleList([
            SpikingResBlock(now_ch, now_ch, tdim),
            SpikingFullySpatialTemporalSelfAttention(now_ch),
            SpikingResBlock(now_ch, now_ch, tdim),
        ])
        # Upsampling
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(c_mult))):
            out_ch = c_base * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(SpikingResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim))
                if i in attn:
                    self.upblocks.append(SpikingFullySpatialTemporalSelfAttention(out_ch))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(SpikingUpSample(now_ch))
        # End
        self.end_bn1 = nn.BatchNorm3d(now_ch)
        self.end_swish1 = Swish()
        self.end_conv1 = nn.Conv3d(now_ch, c_out, kernel_size=(3,3,1), stride=(1,1,1), padding=(1,1,0))
        self.end_conv2 = nn.Conv3d(c_out, c_out, kernel_size=(3,3,1), stride=(1,1,1), padding=(1,1,0))
        self.end_bn2 = nn.BatchNorm3d(c_out)
        self.end_swish2 = Swish()
        self.membrane_output_layer = MembraneOutputLayer(timestep=self.timestep)
        # Init
        functional.set_step_mode(self, step_mode='m')
        self.initialize()
        print(self)

    def initialize(self):
        init.xavier_uniform_(self.begain_conv1.weight)
        init.zeros_(self.begain_conv1.bias)
        init.xavier_uniform_(self.end_conv1.weight, gain=1e-5)
        init.zeros_(self.end_conv1.bias)

    def forward(self, x, t):
        # x: [B, C, H, W] t: [B]
        x = x.unsqueeze(0).repeat(self.timestep, 1, 1, 1, 1)  
        # Timestep embedding
        temb = self.time_embedding(t)
        # Begain
        T, B, C, H, W = x.shape
        h = x.permute(1, 2, 3, 4, 0)  # [B, C, H, W, T]
        h = self.begain_conv1(h)
        h = self.begain_bn1(h).permute(4, 0, 1, 2, 3)  # [T, B, C, H, W]
        h = self.begain_neuron(h)
        h = h.permute(1, 2, 3, 4, 0)  # [B, C, H, W, T]
        h = self.begain_conv2(h)
        h = self.begain_bn2(h).permute(4, 0, 1, 2, 3)  # [T, B, C, H, W]
        # Downsampling
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            if not isinstance(layer, SpikingFullySpatialTemporalSelfAttention):
                hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, SpikingResBlock):
                h = torch.cat([h, hs.pop()], dim=2)
            h = layer(h, temb)
        # End
        T, B, C, H, W = h.shape
        h = h.permute(1, 2, 3, 4, 0)  # [B, C, H, W, T]
        h = self.end_bn1(h)
        h = self.end_swish1(h)
        h = self.end_conv1(h)
        h_temp = h
        h_temp = self.end_conv2(h_temp)
        h_temp = self.end_bn2(h_temp)
        h = self.end_swish2(h_temp) + h
        h = self.membrane_output_layer(h.permute(4, 0, 1, 2, 3)) # [B, C, H, W]
        return h
   
class SpikingDiffusion:    
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=[10,3,32,32], device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size

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
    
    def get_noise(self, x, p = 0.5):
        index = torch.rand_like(x) < p
        noise = torch.ones_like(x)
        noise[index] = -1
        return noise

    def noise_images(self, x, t):
        at = self.compute_alpha(self.beta, t.long())
        noise = torch.randn_like(x)
        noise_image = at.sqrt() * x + (1 - at).sqrt() * noise
        return noise_image, noise
    
    def sample(self, model, device, skip, eta_flag='ddim', n=None):
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
            x = torch.randn_like(x)
            # 去噪
            for (i, j) in zip(reversed(seq), reversed(seq_next)):
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
                x = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * noise
                functional.reset_net(model)
        model.train()
        return x.clamp(-1, 1)
    
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
            x = torch.randn_like(x)
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
                x = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * noise
                functional.reset_net(model)
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    
    def sample_test(self, x, model, device, skip, eta_flag='ddim'):
        forward_xs = []
        backward_xs = []
        seq = range(0, self.noise_steps, skip)
        seq_next = [-1] + list(seq[:-1])
        with torch.no_grad():
            # 加噪
            forward_x = x
            for i in range(1, self.noise_steps, skip):
                t = (torch.ones(x.shape[0]) * i).long().to(device)
                forward_x, forward_Ɛ = self.noise_images(x, t)
                forward_xs.append(forward_x)
                print('t:{} forward_x:{} forward_Ɛ:{}'.format(i, forward_x.sum(), forward_Ɛ.sum()))

            # 去噪
            backward_x = forward_x
            
            for (i, j) in tqdm(zip(reversed(seq), reversed(seq_next))):
                t = (torch.ones(backward_x.shape[0]) * i).long().to(device)
                next_t = (torch.ones(backward_x.shape[0]) * j).long().to(device)
                at = self.compute_alpha(self.beta, t.long())
                at_next = self.compute_alpha(self.beta, next_t.long())
                noise = model(backward_x, t)
                x0_t = (backward_x - noise * (1 - at).sqrt()) / at.sqrt()
                if eta_flag == 'ddim':
                    eta = 0
                else:
                    eta = (1 - at_next) / (1 - at).sqrt() * (1 - at / at_next).sqrt()
                c1 = (
                    eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                backward_x = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * noise
                # functional.reset_net(model)
                backward_xs.append(backward_x)
                print('t:{} backward_x:{} backward_Ɛ:{}'.format(i, backward_x.sum(), noise.sum()))
        return forward_xs, backward_xs
