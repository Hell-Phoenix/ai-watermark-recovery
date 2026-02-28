"""HiDDeN-style watermark encoder.

Architecture (from HiDDeN, Zhu et al. 2018):
  1. Message preparation  — FC + reshape to spatial feature map
  2. Concat                — cover image channels + message feature map
  3. Encoder conv blocks  — extract joint features
  4. Residual blocks      — refine the watermark signal
  5. Final conv           — produce a 3-channel residual image
  6. Output               — cover + scaled residual  →  watermarked image

The residual is added with a learnable strength factor so the network can
control imperceptibility vs. robustness during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Pre-activation ResNet block (BN → ReLU → Conv → BN → ReLU → Conv)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm → ReLU convenience block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HiDDeNEncoder(nn.Module):
    """HiDDeN-style watermark encoder with residual blocks.

    Parameters
    ----------
    message_length : int
        Number of bits in the watermark payload (default 48).
    encoder_channels : int
        Number of feature channels in the encoder body (default 64).
    num_residual_blocks : int
        Number of ResNet residual blocks (default 8).
    residual_strength : float
        Initial scaling factor for the additive residual (default 0.1).
        A smaller value favours imperceptibility at the start of training.
    """

    def __init__(
        self,
        message_length: int = 48,
        encoder_channels: int = 64,
        num_residual_blocks: int = 8,
        residual_strength: float = 0.1,
    ) -> None:
        super().__init__()

        self.message_length = message_length
        self.encoder_channels = encoder_channels

        # ---- Message preparation ------------------------------------------------
        # Project the binary message to a spatial feature map that can be
        # tiled across the image and concatenated with image features.
        self.message_fc = nn.Linear(message_length, encoder_channels)

        # ---- Image feature extraction -------------------------------------------
        # Initial convolutions on the cover image (3 channels)
        self.image_pre = nn.Sequential(
            ConvBNReLU(3, encoder_channels),
            ConvBNReLU(encoder_channels, encoder_channels),
        )

        # ---- Joint processing (image features + message features) ----------------
        # After concat: encoder_channels (image) + encoder_channels (message) = 2x
        self.after_concat = nn.Sequential(
            ConvBNReLU(encoder_channels * 2, encoder_channels),
        )

        # ---- Residual body -------------------------------------------------------
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(encoder_channels) for _ in range(num_residual_blocks)]
        )

        # ---- Output head ---------------------------------------------------------
        # Produce a 3-channel residual (same spatial size as input image)
        self.output_conv = nn.Sequential(
            nn.Conv2d(encoder_channels, 3, kernel_size=1, bias=False),
        )

        # Learnable residual strength — clamped to [0, 1] during forward
        self.strength = nn.Parameter(torch.tensor(residual_strength))

    # ------------------------------------------------------------------
    def forward(
        self,
        image: torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:
        """Embed *message* into *image* and return the watermarked image.

        Parameters
        ----------
        image : Tensor   (B, 3, H, W)  — cover image in [0, 1].
        message : Tensor (B, 48)        — binary message (0/1 floats).

        Returns
        -------
        Tensor (B, 3, H, W) — watermarked image, clamped to [0, 1].
        """
        B, _, H, W = image.shape

        # 1. Image features
        image_features = self.image_pre(image)  # (B, C, H, W)

        # 2. Message → spatial feature map
        msg_feat = self.message_fc(message)  # (B, C)
        msg_feat = msg_feat.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        msg_feat = msg_feat.expand(-1, -1, H, W)  # (B, C, H, W)

        # 3. Concatenate image features + message features
        concat = torch.cat([image_features, msg_feat], dim=1)  # (B, 2C, H, W)

        # 4. Joint processing + residual blocks
        features = self.after_concat(concat)  # (B, C, H, W)
        features = self.residual_blocks(features)  # (B, C, H, W)

        # 5. Produce residual and add to cover image
        residual = self.output_conv(features)  # (B, 3, H, W)
        strength = self.strength.clamp(0.0, 1.0)
        watermarked = image + strength * residual

        return watermarked.clamp(0.0, 1.0)
