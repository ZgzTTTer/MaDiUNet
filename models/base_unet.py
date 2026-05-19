from abc import ABC, abstractmethod
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
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
        return self.timembedding(t)


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        return self.main(x)


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.main(x)


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj = nn.Conv2d(in_ch, in_ch, 1)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W)
        w = torch.bmm(q, k) * (C ** (-0.5))
        w = F.softmax(w, dim=-1)

        v = v.reshape(B, C, H * W).permute(0, 2, 1)
        h = torch.bmm(w, v)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        h = self.proj(h)
        return x + h


class BaseUNet(nn.Module, ABC):
    """Abstract base class for UNet architectures used in diffusion models."""
    
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout, in_channels=2, out_channels=1):
        super().__init__()
        self.T = T
        self.ch = ch
        self.ch_mult = ch_mult
        self.attn = attn
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        tdim = ch * 4
        self.tdim = tdim
        
        # Time embedding
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        
        # Input projection
        self.head = nn.Conv2d(self.in_channels, ch, 3, 1, 1)
        self.initialize_conv(self.head)
        
        # Build encoder
        self.downblocks = nn.ModuleList()
        self.down_channels = [ch]  # Track channels for skip connections
        now_ch = ch
        
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                block = self.create_block(now_ch, out_ch, tdim, dropout, attn=(i in attn))
                self.downblocks.append(block)
                now_ch = out_ch
                self.down_channels.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                self.down_channels.append(now_ch)
        
        # Middle blocks
        self.middleblocks = nn.ModuleList([
            self.create_block(now_ch, now_ch, tdim, dropout, attn=True),
            self.create_block(now_ch, now_ch, tdim, dropout, attn=False),
        ])
        
        # Build decoder
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                # Skip connection: concatenate with encoder features
                in_ch = self.down_channels.pop() + now_ch
                block = self.create_block(in_ch, out_ch, tdim, dropout, attn=(i in attn))
                self.upblocks.append(block)
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        
        # Output projection
        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, self.out_channels, 3, 1, 1)
        )
        self.initialize_conv(self.tail[-1])
    
    @abstractmethod
    def create_block(self, in_ch, out_ch, tdim, dropout, attn=False):
        """Create a block specific to the UNet variant."""
        pass
    
    def initialize_conv(self, module):
        """Initialize convolutional layers."""
        if isinstance(module, nn.Conv2d):
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
    
    def forward(self, x, t):
        """Forward pass through the UNet."""
        # Time embedding
        temb = self.time_embedding(t)
        
        # Input projection
        h = self.head(x)
        hs = [h]  # Store for skip connections
        
        # Encoder
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        
        # Decoder
        for layer in self.upblocks:
            if hasattr(layer, 'forward') and not isinstance(layer, (DownSample, UpSample)):
                # This is a block that needs skip connection
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        
        # Output projection
        h = self.tail(h)
        return h