"""TCN + Causal Transformer for IMU-only speed & heading estimation.

Extracted verbatim from dead-sih-main.ipynb cell 9.
Architecture: (B, 50, 9) IMU window -> TCN (4 dilated causal blocks) -> SinusoidalPE
-> 2-layer causal Transformer -> last-timestep linear head -> (B, 3)
Output: [velocity_normalized, sin(heading_rad), cos(heading_rad)]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D dilated conv padded on the left only — output at t never sees the future."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.norm2 = nn.GroupNorm(1, out_ch)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.drop(F.relu(self.norm1(self.conv1(x))))
        out = self.drop(F.relu(self.norm2(self.conv2(out))))
        return F.relu(out + self.res(x))


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TCN_Transformer(nn.Module):
    def __init__(self, in_features=9, d_model=64, kernel_size=3, dilations=(1, 2, 4, 8),
                 n_heads=4, n_layers=2, dim_ff=128, dropout=0.2, out_features=3, window=50):
        super().__init__()
        blocks, ch = [], in_features
        for d in dilations:
            blocks.append(TCNBlock(ch, d_model, kernel_size, d, dropout))
            ch = d_model
        self.tcn = nn.Sequential(*blocks)

        self.pos = SinusoidalPE(d_model, max_len=window)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(window, window, dtype=torch.bool), diagonal=1),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, out_features))

    def forward(self, x):                                  # (B, 50, 9)
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)    # (B, 50, 64)
        x = self.pos(x)
        T = x.size(1)
        x = self.encoder(x, mask=self.causal_mask[:T, :T])
        return self.head(x[:, -1])                         # (B, 3)
