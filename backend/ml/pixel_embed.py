"""Pixel-domain watermark embedding and extraction.

Wraps the Integer Invertible Watermark Network (iIWN) in a clean
high-level API for pixel-domain watermarking.  This module provides:

  - ``PixelWatermarkConfig`` — configuration dataclass
  - ``PixelWatermarkEmbedder`` — embed(image, message) → watermarked
  - ``PixelWatermarkExtractor`` — extract(watermarked) → predicted bits

The extraction in the absence of the original message is done via a
learned extraction head (blind decoder), since the full iIWN inverse
requires knowledge of the embedded message.  The blind decoder uses
a lightweight CNN to directly predict the embedded bits from the
watermarked image.

References:
  - iIWN architecture: see ``backend/ml/iiwn.py``
  - HiDDeN blind decoder pattern: see ``backend/ml/decoder.py``
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.ml.iiwn import IntegerInvertibleWatermarkNetwork

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PixelWatermarkConfig:
    """Configuration for pixel-domain watermarking.

    Parameters
    ----------
    message_bits : int
        Number of watermark bits to embed (default 48).
    num_blocks : int
        Number of invertible blocks in iIWN (default 4).
    hidden_channels : int
        Feature channels in coupling networks (default 64).
    msg_embed_dim : int
        Message embedding dimension (default 64).
    strength : float
        Watermark perturbation strength (default 0.1).
    detection_threshold : float
        Threshold for considering watermark present (default 0.6).
    """

    message_bits: int = 48
    num_blocks: int = 4
    hidden_channels: int = 64
    msg_embed_dim: int = 64
    strength: float = 0.1
    detection_threshold: float = 0.6


# ---------------------------------------------------------------------------
# Blind extraction head
# ---------------------------------------------------------------------------

class BlindPixelDecoder(nn.Module):
    """Lightweight CNN that predicts watermark bits directly from image.

    Used when we don't know the embedded message (blind extraction).
    The iIWN inverse requires knowing the message, so this network
    learns to extract the bits without that knowledge.

    Parameters
    ----------
    message_bits : int
        Number of bits to predict.
    base_channels : int
        Base feature channels (default 64).
    """

    def __init__(
        self,
        message_bits: int = 48,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.message_bits = message_bits

        self.features = nn.Sequential(
            # Block 1: 3 → base
            nn.Conv2d(3, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            # Block 2: base → 2*base
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            # Block 3: 2*base → 4*base
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            # Block 4: 4*base → 4*base
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(base_channels * 2, message_bits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict watermark bit logits from image.

        Parameters
        ----------
        x : (B, 3, H, W) in [0, 1]

        Returns
        -------
        (B, message_bits) — logits (apply sigmoid for probabilities)
        """
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.classifier(h)


# ---------------------------------------------------------------------------
# High-level embedder
# ---------------------------------------------------------------------------

class PixelWatermarkEmbedder(nn.Module):
    """Pixel-domain watermark embedder using iIWN.

    Parameters
    ----------
    config : PixelWatermarkConfig
        Configuration for the embedder.
    """

    def __init__(self, config: PixelWatermarkConfig | None = None) -> None:
        super().__init__()
        self.config = config or PixelWatermarkConfig()

        self.iiwn = IntegerInvertibleWatermarkNetwork(
            num_blocks=self.config.num_blocks,
            hidden_channels=self.config.hidden_channels,
            message_bits=self.config.message_bits,
            msg_embed_dim=self.config.msg_embed_dim,
            strength=self.config.strength,
        )

    def embed(
        self,
        image: torch.Tensor,
        message: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Embed a watermark message into an image.

        Parameters
        ----------
        image : (B, 3, H, W) in [0, 1]
        message : (B, N) binary {0, 1}

        Returns
        -------
        dict with:
          - ``watermarked``: (B, 3, H, W) watermarked image
          - ``residual``: (B, 3, H, W) pixel-level change (for analysis)
          - ``psnr``: scalar — peak signal-to-noise ratio
        """
        watermarked = self.iiwn.embed(image, message)

        with torch.no_grad():
            residual = watermarked - image
            mse = F.mse_loss(watermarked, image)
            psnr = -10.0 * torch.log10(mse.clamp(min=1e-10))

        return {
            "watermarked": watermarked,
            "residual": residual.detach(),
            "psnr": psnr,
        }

    def forward(
        self,
        image: torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:
        """Embed and return watermarked image only (for training)."""
        return self.iiwn.embed(image, message)


# ---------------------------------------------------------------------------
# High-level extractor
# ---------------------------------------------------------------------------

class PixelWatermarkExtractor(nn.Module):
    """Pixel-domain watermark extractor (blind and informed modes).

    Parameters
    ----------
    config : PixelWatermarkConfig
        Configuration matching the embedder.
    """

    def __init__(self, config: PixelWatermarkConfig | None = None) -> None:
        super().__init__()
        self.config = config or PixelWatermarkConfig()

        # Blind decoder for when message is unknown
        self.blind_decoder = BlindPixelDecoder(
            message_bits=self.config.message_bits,
        )

    def extract_blind(
        self,
        watermarked: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Extract watermark bits without knowing the original message.

        Parameters
        ----------
        watermarked : (B, 3, H, W) in [0, 1]

        Returns
        -------
        dict with:
          - ``logits``: (B, N) raw logits
          - ``predicted_bits``: (B, N) binary {0, 1}
          - ``confidence``: (B,) mean sigmoid probability of predicted bits
          - ``is_watermarked``: (B,) bool — confidence > threshold
        """
        logits = self.blind_decoder(watermarked)
        probs = torch.sigmoid(logits)
        predicted_bits = (probs > 0.5).float()

        # Confidence = mean of max(prob, 1-prob) per sample
        confidence = torch.max(probs, 1.0 - probs).mean(dim=-1)

        is_watermarked = confidence > self.config.detection_threshold

        return {
            "logits": logits,
            "predicted_bits": predicted_bits,
            "confidence": confidence,
            "is_watermarked": is_watermarked,
        }

    def extract(
        self,
        watermarked: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Alias for :meth:`extract_blind` (for DualDomainDetector)."""
        return self.extract_blind(watermarked)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict logits (for training)."""
        return self.blind_decoder(x)


# ---------------------------------------------------------------------------
# Combined training loss for pixel watermarking pipeline
# ---------------------------------------------------------------------------

class PixelWatermarkLoss(nn.Module):
    """Combined loss for training pixel watermark embed + extract.

    L = λ_img · MSE(watermarked, cover)
      + λ_msg · BCE(predicted_logits, message)

    Parameters
    ----------
    lambda_img : float
        Image fidelity weight (default 1.0).
    lambda_msg : float
        Message accuracy weight (default 1.0).
    """

    def __init__(
        self,
        lambda_img: float = 1.0,
        lambda_msg: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_img = lambda_img
        self.lambda_msg = lambda_msg

    def forward(
        self,
        cover: torch.Tensor,
        watermarked: torch.Tensor,
        logits: torch.Tensor,
        message: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Parameters
        ----------
        cover : (B, 3, H, W) — original image
        watermarked : (B, 3, H, W) — embedder output
        logits : (B, N) — extractor output
        message : (B, N) — ground truth bits

        Returns
        -------
        dict with ``loss``, ``img_mse``, ``msg_bce``, ``psnr``, ``bit_acc``.
        """
        img_mse = F.mse_loss(watermarked, cover)
        msg_bce = F.binary_cross_entropy_with_logits(logits, message.float())

        total = self.lambda_img * img_mse + self.lambda_msg * msg_bce

        with torch.no_grad():
            psnr = -10.0 * torch.log10(img_mse.clamp(min=1e-10))
            predicted = (torch.sigmoid(logits) > 0.5).float()
            bit_acc = (predicted == message.float()).float().mean()

        return {
            "loss": total,
            "img_mse": img_mse.detach(),
            "msg_bce": msg_bce.detach(),
            "psnr": psnr,
            "bit_acc": bit_acc,
        }
