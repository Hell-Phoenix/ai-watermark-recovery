"""Loss functions for watermark embedding and recovery training.

This module provides the combined training objectives described in the paper:

*  **WatermarkLoss** — Joint loss for encoder-decoder training:
       L = λ_img · LPIPS + λ_mse · MSE + λ_msg · BCE
   where the image terms enforce imperceptibility and the message term
   enforces robust bit recovery.

*  **RestorationLoss** — Loss for IGRM / restoration network training:
       L = λ_bit · BCE(decoded_bits, gt_bits) + λ_percep · MSE(restored, clean)
   with λ_bit >> λ_percep so the network prioritises **bit structure
   preservation** over visual quality.

*  **FrequencyLoss** — FFT-domain L1 loss on amplitude spectrum, encouraging
   high-frequency watermark signal preservation.

All losses accept (B, …) tensors and return a scalar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Perceptual loss (lightweight, no external VGG dependency)
# ---------------------------------------------------------------------------

class _VGGFeatureExtractor(nn.Module):
    """Minimal multi-scale feature extractor using VGG-style conv blocks.

    This avoids downloading full VGG weights while still providing a
    reasonable perceptual signal for training.  When pretrained VGG is
    available the user can swap this out for ``torchvision.models.vgg16``.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        # Three feature extraction stages at 1×, ½×, ¼× resolution
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return [f1, f2, f3]


class LPIPSLoss(nn.Module):
    """Learned Perceptual Image Patch Similarity (lightweight variant).

    Computes L1 distance between multi-scale feature maps of two images,
    providing a differentiable perceptual similarity signal.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = _VGGFeatureExtractor()
        # Freeze feature extractor weights (train on random init features
        # as a fixed transform — standard practice for lightweight LPIPS)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perceptual distance between images *x* and *y*.

        Parameters
        ----------
        x, y : Tensor (B, 3, H, W) in [0, 1]

        Returns
        -------
        Scalar tensor — mean perceptual distance.
        """
        feats_x = self.feature_extractor(x)
        feats_y = self.feature_extractor(y)
        loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for fx, fy in zip(feats_x, feats_y):
            # Normalise features per-channel before comparing
            fx_norm = fx / (fx.norm(dim=1, keepdim=True) + 1e-8)
            fy_norm = fy / (fy.norm(dim=1, keepdim=True) + 1e-8)
            loss = loss + F.l1_loss(fx_norm, fy_norm)
        return loss / len(feats_x)


# ---------------------------------------------------------------------------
# Frequency-domain loss
# ---------------------------------------------------------------------------

class FrequencyLoss(nn.Module):
    """FFT amplitude-spectrum L1 loss.

    Encourages preservation of high-frequency watermark signal components
    that are most susceptible to JPEG / blur attacks.

    Parameters
    ----------
    weight_high_freq : float
        Extra weighting factor applied to frequency bins in the outer 50%
        of the spectrum radius, default 2.0.
    """

    def __init__(self, weight_high_freq: float = 2.0) -> None:
        super().__init__()
        self.weight_high_freq = weight_high_freq
        self._mask: torch.Tensor | None = None

    def _get_freq_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Create a radial mask that up-weights high frequencies."""
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=device).float() - cy
        x = torch.arange(W, device=device).float() - cx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        radius = torch.sqrt(xx**2 + yy**2)
        max_r = math.sqrt(cy**2 + cx**2)
        # Frequencies beyond 50% of max radius get extra weight
        mask = torch.where(radius > max_r * 0.5, self.weight_high_freq, 1.0)
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Frequency-domain L1 loss.

        Parameters
        ----------
        x, y : Tensor (B, C, H, W)

        Returns
        -------
        Scalar tensor.
        """
        # 2D FFT → shift DC to centre → amplitude spectrum
        fft_x = torch.fft.fft2(x, norm="ortho")
        fft_y = torch.fft.fft2(y, norm="ortho")
        amp_x = torch.fft.fftshift(fft_x).abs()
        amp_y = torch.fft.fftshift(fft_y).abs()

        _, _, H, W = x.shape
        mask = self._get_freq_mask(H, W, x.device)
        return (mask * (amp_x - amp_y).abs()).mean()


# ---------------------------------------------------------------------------
# Combined loss: Encoder-Decoder watermarking
# ---------------------------------------------------------------------------

@dataclass
class WatermarkLossConfig:
    """Weights for the combined watermarking loss."""

    lambda_lpips: float = 1.0
    lambda_mse: float = 0.5
    lambda_msg: float = 10.0


class WatermarkLoss(nn.Module):
    """Combined loss for encoder-decoder watermark training.

    L = λ_lpips · LPIPS(cover, wm) + λ_mse · MSE(cover, wm)
      + λ_msg · BCE(decoded_bits, gt_bits)

    Parameters
    ----------
    config : WatermarkLossConfig
        Loss weighting configuration.
    """

    def __init__(self, config: WatermarkLossConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or WatermarkLossConfig()
        self.lpips = LPIPSLoss()

    def forward(
        self,
        cover: torch.Tensor,
        watermarked: torch.Tensor,
        decoded_logits: torch.Tensor,
        gt_bits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Parameters
        ----------
        cover : (B, 3, H, W) original image
        watermarked : (B, 3, H, W) watermarked image
        decoded_logits : (B, N) raw logits from decoder
        gt_bits : (B, N) ground-truth bit values in {0, 1}

        Returns
        -------
        dict with keys: ``loss`` (total), ``lpips``, ``mse``, ``msg_bce``,
        ``bit_acc``.
        """
        l_lpips = self.lpips(cover, watermarked)
        l_mse = F.mse_loss(watermarked, cover)
        l_msg = F.binary_cross_entropy_with_logits(decoded_logits, gt_bits.float())

        total = (
            self.cfg.lambda_lpips * l_lpips
            + self.cfg.lambda_mse * l_mse
            + self.cfg.lambda_msg * l_msg
        )

        with torch.no_grad():
            bit_acc = ((decoded_logits > 0).float() == gt_bits.float()).float().mean()

        return {
            "loss": total,
            "lpips": l_lpips.detach(),
            "mse": l_mse.detach(),
            "msg_bce": l_msg.detach(),
            "bit_acc": bit_acc,
        }


# ---------------------------------------------------------------------------
# Combined loss: IGRM restoration
# ---------------------------------------------------------------------------

@dataclass
class RestorationLossConfig:
    """Weights for the IGRM restoration loss.

    ``lambda_bit`` is intentionally much larger than ``lambda_percep``
    so the restoration network focuses on preserving watermark bit
    structure rather than visual quality.
    """

    lambda_bit: float = 50.0
    lambda_percep: float = 1.0
    lambda_freq: float = 5.0


class RestorationLoss(nn.Module):
    """Loss for IGRM training — prioritises bit-recovery accuracy.

    L = λ_bit · BCE(decoder(restored), gt_bits)
      + λ_percep · MSE(restored, clean_wm)
      + λ_freq · FreqLoss(restored, clean_wm)

    The decoder must be provided at construction time and is kept **frozen**
    during IGRM training (gradients flow through it but its weights are not
    updated in the IGRM optimizer).

    Parameters
    ----------
    decoder : nn.Module
        Frozen watermark decoder (logits output).
    config : RestorationLossConfig
        Loss weight configuration.
    """

    def __init__(
        self,
        decoder: nn.Module,
        config: RestorationLossConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or RestorationLossConfig()
        self.decoder = decoder
        self.freq_loss = FrequencyLoss(weight_high_freq=2.0)

        # Freeze decoder — it provides gradient signal but is not optimized
        for param in self.decoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        restored: torch.Tensor,
        clean_watermarked: torch.Tensor,
        gt_bits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute restoration loss.

        Parameters
        ----------
        restored : (B, 3, H, W) — IGRM output
        clean_watermarked : (B, 3, H, W) — original watermarked image (no attack)
        gt_bits : (B, N) — ground-truth embedded bits

        Returns
        -------
        dict with keys: ``loss``, ``bit_bce``, ``percep_mse``, ``freq``,
        ``bit_acc``.
        """
        # Bit-recovery loss through frozen decoder
        decoded_logits = self.decoder(restored)
        l_bit = F.binary_cross_entropy_with_logits(decoded_logits, gt_bits.float())

        # Perceptual MSE to clean watermarked image
        l_percep = F.mse_loss(restored, clean_watermarked)

        # Frequency domain preservation
        l_freq = self.freq_loss(restored, clean_watermarked)

        total = (
            self.cfg.lambda_bit * l_bit
            + self.cfg.lambda_percep * l_percep
            + self.cfg.lambda_freq * l_freq
        )

        with torch.no_grad():
            bit_acc = ((decoded_logits > 0).float() == gt_bits.float()).float().mean()

        return {
            "loss": total,
            "bit_bce": l_bit.detach(),
            "percep_mse": l_percep.detach(),
            "freq": l_freq.detach(),
            "bit_acc": bit_acc,
        }
