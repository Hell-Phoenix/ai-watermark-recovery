"""Inverse Generative Restoration Module (IGRM).

Restores watermark bit structure from degraded / attacked images.
Unlike super-resolution networks that optimise for visual quality, IGRM is
trained to maximise **bit-recovery accuracy** (BER → 0).

Architecture:
  - U-Net with skip connections at every level
  - Encoder: strided convolutions (downsample 4×)
  - Bottleneck: residual blocks
  - Decoder: transposed convolutions + skip-add from encoder
  - Output: 3-channel restored image (same spatial size as input)

Loss weighting:   L = λ_bit · BCE(decoded_bits, gt_bits) + λ_percep · MSE
where λ_bit >> λ_percep, so the network focuses on preserving the embedded
watermark signal rather than visual fidelity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Conv → BN → ReLU × 2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DownBlock(nn.Module):
    """Strided conv downsample → ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2, bias=False)
        self.conv = _ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.down(x))


class _UpBlock(nn.Module):
    """Transposed conv upsample → concat skip → ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.conv = _ConvBlock(out_ch * 2, out_ch)  # *2 for skip concat

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd spatial sizes
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class RestorationBlock(nn.Module):
    """Bottleneck residual block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class IGRM(nn.Module):
    """Inverse Generative Restoration Module.

    Takes a degraded watermarked image and outputs a restored image
    optimised for watermark bit recovery.

    Parameters
    ----------
    in_channels : int
        Input image channels (default: 3).
    base_channels : int
        Feature channels at the first level (default: 64).
    depth : int
        Number of encoder / decoder levels (default: 4).
    num_bottleneck_blocks : int
        Number of residual blocks in the bottleneck (default: 4).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        depth: int = 4,
        num_bottleneck_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.depth = depth

        # ---- Encoder ----
        ch = base_channels
        self.enc_first = _ConvBlock(in_channels, ch)
        self.encoders = nn.ModuleList()
        for _ in range(depth - 1):
            self.encoders.append(_DownBlock(ch, ch * 2))
            ch *= 2

        # ---- Bottleneck ----
        self.bottleneck = nn.Sequential(
            *[RestorationBlock(ch) for _ in range(num_bottleneck_blocks)]
        )

        # ---- Decoder ----
        self.decoders = nn.ModuleList()
        for _ in range(depth - 1):
            self.decoders.append(_UpBlock(ch, ch // 2))
            ch //= 2

        # ---- Output ----
        self.output_conv = nn.Conv2d(ch, in_channels, kernel_size=1)

    def forward(self, degraded: torch.Tensor) -> torch.Tensor:
        """Restore watermark structure from a degraded image.

        Parameters
        ----------
        degraded : Tensor (B, 3, H, W) — attacked / degraded image in [0, 1].

        Returns
        -------
        Tensor (B, 3, H, W) — restored image in [0, 1].
        """
        # Encoder (collect skip connections)
        skips: list[torch.Tensor] = []
        x = self.enc_first(degraded)
        skips.append(x)
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder (consume skips in reverse, skip the last one which is bottleneck input)
        skips = skips[:-1]  # drop the deepest skip (already in x)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        # Output: residual learning — predict the "clean" watermarked image
        restored = self.output_conv(x) + degraded
        return restored.clamp(0.0, 1.0)
