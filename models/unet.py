from models.base_unet import BaseUNet
from models.blocks import VanillaBlock, ResBlock, MambaBlock, DenseBlock, SwinBlock
from models.base_unet import TimeEmbedding, DownSample, UpSample, Swish
from torch import nn
from torch.nn import init
import torch

class VanillaUNet(BaseUNet):
    """Vanilla UNet with simple convolutional blocks."""
    
    def create_block(self, in_ch, out_ch, tdim, dropout, attn=False):
        return VanillaBlock(in_ch, out_ch, tdim, dropout, attn)


class ResUNet(BaseUNet):
    """ResUNet with residual blocks."""
    
    def create_block(self, in_ch, out_ch, tdim, dropout, attn=False):
        return ResBlock(in_ch, out_ch, tdim, dropout, attn)


class MambaUNet(BaseUNet):
    """MambaUNet with Mamba blocks for sequence modeling."""
    
    def create_block(self, in_ch, out_ch, tdim, dropout, attn=False):
        return MambaBlock(in_ch, out_ch, tdim, dropout, attn)


class DenseUNet(BaseUNet):
    """DenseUNet with densely connected blocks throughout the entire network."""
    
    def create_block(self, in_ch, out_ch, tdim, dropout, attn=False):
        return DenseBlock(in_ch, out_ch, tdim, dropout, attn)


class SwinUNet(BaseUNet):
    """SwinUNet with Swin Transformer blocks in encoder and ResBlocks in decoder."""
    
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout, in_channels=2, out_channels=1):
        # 不调用 BaseUNet 的 __init__，而是手动初始化
        super(BaseUNet, self).__init__()  # 只调用 nn.Module 的 __init__
        
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
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        
        # Build encoder with SwinBlocks
        self.downblocks = nn.ModuleList()
        self.down_channels = [ch]
        now_ch = ch
        
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                block = SwinBlock(now_ch, out_ch, tdim, dropout, attn=(i in attn))
                self.downblocks.append(block)
                now_ch = out_ch
                self.down_channels.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                self.down_channels.append(now_ch)
        
        # Middle blocks with SwinBlocks
        self.middleblocks = nn.ModuleList([
            SwinBlock(now_ch, now_ch, tdim, dropout, attn=True),
            SwinBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])
        
        # Build decoder with ResBlocks
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                in_ch = self.down_channels.pop() + now_ch
                block = ResBlock(in_ch, out_ch, tdim, dropout, attn=(i in attn))
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
        init.xavier_uniform_(self.tail[-1].weight)
        init.zeros_(self.tail[-1].bias)
    
    def forward(self, x, t):
        """Forward pass - 复用 BaseUNet 的 forward 逻辑"""
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
    
    def create_block(self, in_ch, out_ch, tdim, dropout, attn=False):
        # 这个方法在 SwinUNet 中不会被调用
        pass