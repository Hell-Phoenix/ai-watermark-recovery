"""Enhanced U-Net for conditional watermark restoration.

This extends the basic IGRM U-Net (``igrm.py``) with:

1. **Timestep / severity conditioning** — sinusoidal positional embedding
   projected to each block, enabling the network to adapt its restoration
   strategy based on the estimated attack severity.

2. **Self-attention at the bottleneck** — multi-head self-attention captures
   long-range spatial dependencies critical for recovering watermark signals
   from large crops or heavy geometric distortion.

3. **Channel attention (SE blocks)** — lightweight squeeze-and-excitation
   after each decoder block to focus on channels carrying watermark signal.

4. **Residual-in-residual structure** — deeper feature propagation without
   vanishing gradients.

The module is designed to be trained with ``RestorationLoss`` from
``losses.py`` where λ_bit >> λ_percep.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding (same style as DDPM / diffusion models)
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal positional embedding for timestep *t*.

    Parameters
    ----------
    t : Tensor (B,) — timestep or severity scalar in [0, 1].
    dim : int — embedding dimension (must be even).

    Returns
    -------
    Tensor (B, dim) — positional embedding.
    """
    assert dim % 2 == 0, "Embedding dim must be even"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Recalibrates channel-wise feature responses by modelling
    inter-channel dependencies.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class ConditionalConvBlock(nn.Module):
    """Conv → GroupNorm → SiLU × 2, conditioned on a timestep embedding.

    The timestep embedding is projected to a per-channel scale+shift
    applied after the first group normalisation (adaptive group norm).
    """

    def __init__(self, in_ch: int, out_ch: int, t_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)

        # Timestep conditioning: project to scale & shift
        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, out_ch * 2),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn1(self.conv1(x)))

        # Adaptive group norm: scale and shift from timestep embedding
        scale_shift = self.t_proj(t_emb)  # (B, 2*out_ch)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = self.act(self.gn2(self.conv2(h)))
        return h


class CondDownBlock(nn.Module):
    """Strided conv downsample → ConditionalConvBlock."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2, bias=False)
        self.conv = ConditionalConvBlock(in_ch, out_ch, t_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.conv(self.down(x), t_emb)


class CondUpBlock(nn.Module):
    """Transposed conv upsample → concat skip → ConditionalConvBlock + SE."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.conv = ConditionalConvBlock(out_ch * 2, out_ch, t_dim)  # *2 for skip concat
        self.se = SEBlock(out_ch)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        x = self.up(x)
        # Handle odd spatial sizes
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x, t_emb)
        return self.se(x)


class SpatialSelfAttention(nn.Module):
    """Multi-head self-attention on flattened spatial positions.

    Applied at the bottleneck to capture long-range dependencies
    between distant watermark patches.

    Parameters
    ----------
    channels : int
        Feature map channels.
    num_heads : int
        Number of attention heads (default 4).
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + self.proj(h)


class CondResidualBlock(nn.Module):
    """Bottleneck residual block with timestep conditioning."""

    def __init__(self, channels: int, t_dim: int) -> None:
        super().__init__()
        self.conv = ConditionalConvBlock(channels, channels, t_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x, t_emb)


# ---------------------------------------------------------------------------
# Main module: Conditional Restoration U-Net
# ---------------------------------------------------------------------------

class ConditionalRestorationUNet(nn.Module):
    """Enhanced U-Net for watermark restoration with severity conditioning.

    Key improvements over the basic IGRM U-Net:
      - Sinusoidal timestep/severity embedding → adaptive group norm
      - Self-attention at bottleneck for long-range spatial reasoning
      - SE channel attention in decoder for watermark-signal focus
      - Residual learning (output = predicted_residual + input)

    Parameters
    ----------
    in_channels : int
        Input image channels (default 3).
    out_channels : int or None
        Output channels. If *None*, defaults to ``in_channels`` and
        residual learning is used (output = input + predicted_residual).
        When explicitly set to a different value, residual learning is
        disabled (useful for noise prediction denoiser heads).
    base_channels : int
        Feature channels at level 0 (default 64).
    depth : int
        Number of encoder/decoder levels (default 4).
    num_bottleneck_blocks : int
        Number of residual blocks in the bottleneck (default 4).
    t_dim : int
        Timestep embedding dimension (default 256).
    attn_heads : int
        Number of attention heads at the bottleneck (default 4).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int | None = None,
        base_channels: int = 64,
        depth: int = 4,
        num_bottleneck_blocks: int = 4,
        t_dim: int = 256,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.t_dim = t_dim
        self._out_channels = out_channels if out_channels is not None else in_channels
        self._use_residual = (out_channels is None) or (out_channels == in_channels)

        # ---- Timestep embedding MLP ----
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        # ---- Encoder ----
        ch = base_channels
        self.enc_first = ConditionalConvBlock(in_channels, ch, t_dim)
        self.encoders = nn.ModuleList()
        for _ in range(depth - 1):
            self.encoders.append(CondDownBlock(ch, ch * 2, t_dim))
            ch *= 2

        # ---- Bottleneck ----
        bottleneck_layers: list[nn.Module] = []
        for i in range(num_bottleneck_blocks):
            bottleneck_layers.append(CondResidualBlock(ch, t_dim))
            # Insert self-attention in the middle of the bottleneck
            if i == num_bottleneck_blocks // 2:
                bottleneck_layers.append(SpatialSelfAttention(ch, attn_heads))
        self.bottleneck = nn.ModuleList(bottleneck_layers)

        # ---- Decoder ----
        self.decoders = nn.ModuleList()
        for _ in range(depth - 1):
            self.decoders.append(CondUpBlock(ch, ch // 2, t_dim))
            ch //= 2

        # ---- Output head ----
        self.output_conv = nn.Sequential(
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, self._out_channels, kernel_size=1),
        )

    def forward(
        self,
        degraded: torch.Tensor,
        severity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Restore watermark structure from a degraded image.

        Parameters
        ----------
        degraded : Tensor (B, 3, H, W) — attacked image in [0, 1].
        severity : Tensor (B,) — attack severity in [0, 1], optional.
            If None, assumes severity = 0.5 (mid-range).

        Returns
        -------
        Tensor (B, 3, H, W) — restored image in [0, 1].
        """
        B = degraded.shape[0]

        # Timestep embedding
        if severity is None:
            severity = torch.full((B,), 0.5, device=degraded.device)
        t_emb = sinusoidal_embedding(severity * 1000.0, self.t_dim)  # scale to [0, 1000]
        t_emb = self.t_mlp(t_emb)

        # Encoder (collect skip connections)
        skips: list[torch.Tensor] = []
        x = self.enc_first(degraded, t_emb)
        skips.append(x)
        for enc in self.encoders:
            x = enc(x, t_emb)
            skips.append(x)

        # Bottleneck
        for layer in self.bottleneck:
            if isinstance(layer, CondResidualBlock):
                x = layer(x, t_emb)
            else:
                # SpatialSelfAttention — no timestep conditioning
                x = layer(x)

        # Decoder (consume skips in reverse, drop the deepest)
        skips = skips[:-1]
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip, t_emb)

        # Output projection
        out = self.output_conv(x)

        if self._use_residual:
            # Residual learning: predict correction, add to input
            out = degraded + out
            out = out.clamp(0.0, 1.0)

        return out
