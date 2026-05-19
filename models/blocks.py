import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from models.base_unet import Swish, AttnBlock

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available. MambaBlock will not work.")


class VanillaBlock(nn.Module):
    """Simple convolutional block for Vanilla UNet."""
    
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.activation = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout2d(dropout)
        
        # Time embedding projection
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        
        # Skip connection
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
        # Attention
        self.attn = AttnBlock(out_ch) if attn else nn.Identity()
        
        self.initialize()
    
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
    
    def forward(self, x, temb):
        residual = self.shortcut(x)
        
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        
        # Add time embedding
        h = h + self.temb_proj(temb)[:, :, None, None]
        
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        
        h = h + residual
        
        # Attention
        h = self.attn(h)
        
        return h


class ResBlock(nn.Module):
    """Residual block for ResUNet."""
    
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn = AttnBlock(out_ch) if attn else nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class MambaBlock(nn.Module):
    """Mamba block for MambaUNet."""
    
    def __init__(self, in_ch, out_ch, tdim, dropout=0.3, attn=False):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm is required for MambaBlock")
            
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.proj_in = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.mamba = Mamba(d_model=out_ch)
        self.proj_out = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.temb_proj = nn.Sequential(Swish(), nn.Linear(tdim, out_ch))
        self.attn = AttnBlock(out_ch) if attn else nn.Identity()

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb):
        h = self.norm1(x)
        h = self.proj_in(h)

        B, C, H, W = h.shape
        h = h.permute(0, 2, 3, 1).reshape(B, H * W, C)
        h = self.mamba(h)
        h = h.reshape(B, H, W, C).permute(0, 3, 1, 2)

        h += self.temb_proj(temb)[:, :, None, None]
        h = self.dropout(h)
        h = self.proj_out(h)
        h = self.norm2(h)
        h = self.attn(h)

        return h + self.shortcut(x)


class DenseBlock(nn.Module):
    """Dense block for DenseUNet."""
    
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False, growth_rate=32, num_layers=4):
        super().__init__()
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        
        # Dense layers
        self.dense_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_ch = in_ch + i * growth_rate
            self.dense_layers.append(self._make_dense_layer(layer_in_ch, growth_rate, dropout))
        
        # Transition layer
        total_ch = in_ch + num_layers * growth_rate
        self.transition = nn.Sequential(
            nn.GroupNorm(32, total_ch),
            Swish(),
            nn.Conv2d(total_ch, out_ch, 1),
            nn.Dropout2d(dropout)
        )
        
        # Time embedding projection
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        
        # Skip connection
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
        # Attention
        self.attn = AttnBlock(out_ch) if attn else nn.Identity()
        
        self.initialize()
    
    def _make_dense_layer(self, in_ch, growth_rate, dropout):
        return nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, 4 * growth_rate, 1),
            nn.GroupNorm(32, 4 * growth_rate),
            Swish(),
            nn.Conv2d(4 * growth_rate, growth_rate, 3, 1, 1),
            nn.Dropout2d(dropout)
        )
    
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
    
    def forward(self, x, temb):
        features = [x]
        
        # Dense connections
        for layer in self.dense_layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        
        # Transition
        h = self.transition(torch.cat(features, dim=1))
        
        # Add time embedding
        h += self.temb_proj(temb)[:, :, None, None]
        
        # Skip connection
        h = h + self.shortcut(x)
        
        # Attention
        h = self.attn(h)
        
        return h


class SwinBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False, window_size=8, num_heads=8):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads

        self.input_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )

        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)

        self.attn_layer = nn.MultiheadAttention(out_ch, num_heads, dropout=dropout, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(out_ch, 4 * out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * out_ch, out_ch),
            nn.Dropout(dropout),
        )

        self.extra_attn = AttnBlock(out_ch) if attn else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def window_partition(self, x):
        B, H, W, C = x.shape
        ws = self.window_size
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(-1, ws, ws, C)

    def window_reverse(self, windows, H, W):
        ws = self.window_size
        B = int(windows.shape[0] / ((H // ws) * (W // ws)))
        x = windows.view(B, H // ws, W // ws, ws, ws, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B, H, W, -1)

    def forward(self, x, temb):

        x_res = self.shortcut(x)

        h = self.input_proj(x)

        te = self.temb_proj(temb)[:, :, None, None]
        h = h + te

        h = h.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = h.shape 

        windows = self.window_partition(h)
        windows = windows.view(-1, C, self.window_size * self.window_size).permute(0, 2, 1).contiguous()

        wn = self.norm1(windows)
        attn_out, _ = self.attn_layer(wn, wn, wn)
        windows = windows + attn_out

        wn2 = self.norm2(windows)
        windows = windows + self.mlp(wn2)

        windows = windows.permute(0, 2, 1).contiguous()
        windows = windows.view(-1, self.window_size, self.window_size, C)
        h = self.window_reverse(windows, H, W)

        h = h.permute(0, 3, 1, 2).contiguous()

        h = h + x_res
        h = self.extra_attn(h)
        return h