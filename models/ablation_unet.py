import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from models.base_unet import BaseUNet, Swish, DownSample, UpSample, TimeEmbedding, AttnBlock
from models.blocks import VanillaBlock, ResBlock, MambaBlock, DenseBlock, SwinBlock


class AblationUNet(nn.Module):
    """
    Ablation UNet: selectively replace ResBlocks with MambaBlocks in encoder, middle, and decoder.

    Args:
        T: int, number of diffusion timesteps
        ch: int, base channel count
        ch_mult: list of int, channel multipliers per stage
        attn: list of stage indices for attention
        num_res_blocks: int, number of blocks per stage
        dropout: float, dropout rate
        in_channels: int, input channels
        out_channels: int, output channels
        use_mamba_encoder: bool, replace all encoder blocks
        use_mamba_middle: bool, replace middle blocks
        use_mamba_decoder: bool, replace all decoder blocks
    """
    def __init__(
        self,
        T,
        ch,
        ch_mult,
        attn,
        num_res_blocks,
        dropout,
        in_channels=2,
        out_channels=1,
        use_mamba_encoder=False,
        use_mamba_middle=False,
        use_mamba_decoder=False,
    ):
        super().__init__()

        # Store hyperparameters
        self.T = T
        self.ch = ch
        self.ch_mult = ch_mult
        self.attn = attn
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.tdim = ch * 4

        # Ablation flags
        self.use_mamba_encoder = use_mamba_encoder
        self.use_mamba_middle = use_mamba_middle
        self.use_mamba_decoder = use_mamba_decoder

        # Build network
        self.time_embedding = TimeEmbedding(T, ch, self.tdim)
        self.head = nn.Conv2d(in_channels, ch, 3, 1, 1)
        self.initialize_conv(self.head)

        # Encoder
        self.downblocks = nn.ModuleList()
        self.down_channels = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                Block = MambaBlock if use_mamba_encoder else ResBlock
                self.downblocks.append(Block(now_ch, out_ch, self.tdim, dropout, attn=(i in attn)))
                now_ch = out_ch
                self.down_channels.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                self.down_channels.append(now_ch)

        # Middle blocks
        self.middleblocks = nn.ModuleList([
            (MambaBlock if use_mamba_middle else ResBlock)(now_ch, now_ch, self.tdim, dropout, attn=True),
            (MambaBlock if use_mamba_middle else ResBlock)(now_ch, now_ch, self.tdim, dropout, attn=False),
        ])

        # Decoder
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                in_ch = self.down_channels.pop() + now_ch
                Block = MambaBlock if use_mamba_decoder else ResBlock
                self.upblocks.append(Block(in_ch, out_ch, self.tdim, dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))

        # Output
        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, out_channels, 3, 1, 1)
        )
        self.initialize_conv(self.tail[-1])

    def initialize_conv(self, module):
        if isinstance(module, nn.Conv2d):
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)

    def create_block(self, in_ch, out_ch, tdim, dropout, attn=False):
    
        pass

    def forward(self, x, t):
        """Standard UNet forward: encoder -> middle -> decoder"""
        temb = self.time_embedding(t)
        h = self.head(x)
        hs = [h]

        # Encoder
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)

        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)

        # Decoder
        for layer in self.upblocks:
            if isinstance(layer, (ResBlock, MambaBlock)):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
            h = layer(h, temb)

        return self.tail(h)