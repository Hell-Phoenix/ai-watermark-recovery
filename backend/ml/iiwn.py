"""Integer Invertible Watermark Network (iIWN) — RealNVP normalizing flows.

Reversible pixel-domain watermarking using a normalizing-flow architecture.
The forward pass embeds a binary payload into an image; the learned blind
extraction head recovers the payload from the watermarked image without
knowing the original payload (blind extraction).

Architecture:
  - **Message encoder** — MLP that maps binary payload → continuous
    embedding vector for FiLM conditioning.
  - **Affine coupling layers** — RealNVP-style split-and-transform
    blocks conditioned on the message embedding.  Each block applies
    s,t = NN(x₁, msg_emb); x₂' = x₂ ⊙ exp(s) + t.
  - **Invertible blocks** — two coupling layers with channel swap.
  - **Blind decoder** — lightweight CNN that predicts the embedded
    payload directly from the watermarked image (no message required).

API:
  - ``embed(image, payload)``   → watermarked image  (B, 3, H, W)
  - ``extract(watermarked_image)`` → dict with logits, predicted_bits,
    confidence, is_watermarked
  - ``forward(image, payload)`` → watermarked image  (alias for embed)

References:
  - RealNVP: Dinh et al., "Density estimation using Real-valued
    Non-Volume Preserving transformations", ICLR 2017
  - iIWN / INN watermarking: Lu et al., 2021
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Scale / translate prediction network (with FiLM conditioning)
# ---------------------------------------------------------------------------

class _STNetwork(nn.Module):
    """Predict scale (s) and translate (t) for an affine coupling layer.

    Uses 3-layer CNN with FiLM (Feature-wise Linear Modulation)
    conditioning on the message embedding.

    Parameters
    ----------
    in_channels : int
        Number of input channels (half of total).
    hidden_channels : int
        Feature channels in intermediate layers.
    msg_dim : int
        Message embedding dimension for FiLM modulation.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        msg_dim: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        # FiLM modulation layers
        self.film_scale = nn.Linear(msg_dim, hidden_channels)
        self.film_shift = nn.Linear(msg_dim, hidden_channels)

        # Output heads for scale and translate
        self.out_s = nn.Conv2d(hidden_channels, in_channels, 1)
        self.out_t = nn.Conv2d(hidden_channels, in_channels, 1)

        # Initialise output to near-identity (s≈0, t≈0)
        nn.init.zeros_(self.out_s.weight)
        nn.init.zeros_(self.out_s.bias)
        nn.init.zeros_(self.out_t.weight)
        nn.init.zeros_(self.out_t.bias)

    def forward(
        self,
        x: torch.Tensor,
        msg_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict (s, t) from input features x and message embedding.

        Parameters
        ----------
        x : (B, in_channels, H, W)
        msg_emb : (B, msg_dim)

        Returns
        -------
        s, t : each (B, in_channels, H, W)
        """
        h = self.net(x)

        # FiLM modulation
        gamma = self.film_scale(msg_emb).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.film_shift(msg_emb).unsqueeze(-1).unsqueeze(-1)
        h = h * (1 + gamma) + beta

        s = self.out_s(h)
        t = self.out_t(h)

        # Clamp scale to prevent numerical instability
        s = torch.clamp(s, -2.0, 2.0)

        return s, t


# ---------------------------------------------------------------------------
# Affine coupling layer
# ---------------------------------------------------------------------------

class AffineCouplingLayer(nn.Module):
    """RealNVP-style affine coupling layer with message conditioning.

    Forward:  x1' = x1, x2' = x2 ⊙ exp(s(x1, msg)) + t(x1, msg)
    Inverse:  x1 = x1', x2 = (x2' - t(x1', msg)) ⊙ exp(-s(x1', msg))

    Parameters
    ----------
    channels : int
        Total number of channels (split in half internally).
    hidden_channels : int
        Feature channels in the s/t prediction network.
    msg_dim : int
        Message embedding dimension.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 64,
        msg_dim: int = 64,
    ) -> None:
        super().__init__()
        half_ch = channels // 2
        self.half_ch = half_ch
        self.st_net = _STNetwork(half_ch, hidden_channels, msg_dim)

    def forward(
        self,
        x: torch.Tensor,
        msg_emb: torch.Tensor,
        reverse: bool = False,
    ) -> torch.Tensor:
        """Apply (or invert) the coupling layer.

        Parameters
        ----------
        x : (B, C, H, W)
        msg_emb : (B, msg_dim)
        reverse : bool — if True, apply the inverse transform

        Returns
        -------
        (B, C, H, W)
        """
        x1, x2 = x[:, : self.half_ch], x[:, self.half_ch :]
        s, t = self.st_net(x1, msg_emb)

        if reverse:
            x2 = (x2 - t) * torch.exp(-s)
        else:
            x2 = x2 * torch.exp(s) + t

        return torch.cat([x1, x2], dim=1)


class InvertibleBlock(nn.Module):
    """Two coupling layers with channel swap in between.

    This ensures both channel halves are transformed:
      Layer 1: transforms x2 conditioned on x1
      Swap:    (x1, x2) → (x2, x1)
      Layer 2: transforms (new x2 = old x1) conditioned on (new x1 = old x2)
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 64,
        msg_dim: int = 64,
    ) -> None:
        super().__init__()
        self.coupling1 = AffineCouplingLayer(channels, hidden_channels, msg_dim)
        self.coupling2 = AffineCouplingLayer(channels, hidden_channels, msg_dim)

    def forward(
        self,
        x: torch.Tensor,
        msg_emb: torch.Tensor,
        reverse: bool = False,
    ) -> torch.Tensor:
        """Apply (or invert) both coupling layers with channel swap.

        Parameters
        ----------
        x : (B, C, H, W)
        msg_emb : (B, msg_dim)
        reverse : bool

        Returns
        -------
        (B, C, H, W)
        """
        if reverse:
            # Reverse order and swap
            x = self.coupling2(x, msg_emb, reverse=True)
            x = self._swap_halves(x)
            x = self.coupling1(x, msg_emb, reverse=True)
        else:
            x = self.coupling1(x, msg_emb, reverse=False)
            x = self._swap_halves(x)
            x = self.coupling2(x, msg_emb, reverse=False)
        return x

    @staticmethod
    def _swap_halves(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[1] // 2
        return torch.cat([x[:, half:], x[:, :half]], dim=1)


# ---------------------------------------------------------------------------
# Channel expansion (3ch → 4ch and back)
# ---------------------------------------------------------------------------

class _ChannelPad(nn.Module):
    """Pad 3-channel input to even number of channels (4) for coupling."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pad = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=1)

    def remove(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :3]


# ---------------------------------------------------------------------------
# Blind extraction head
# ---------------------------------------------------------------------------

class _BlindDecoder(nn.Module):
    """Lightweight CNN that predicts watermark bits directly from image.

    Used for blind extraction (no knowledge of the embedded payload
    required).  Processes the watermarked image through a series of
    strided convolutions followed by global average pooling and a
    linear head.

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

        # Global average pooling + linear head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels * 2, message_bits),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict bit logits from watermarked image.

        Parameters
        ----------
        x : (B, 3, H, W) — watermarked image in [0, 1]

        Returns
        -------
        (B, message_bits) — raw logits; positive → bit 1, negative → bit 0.
        """
        features = self.features(x)
        return self.head(features)


# ---------------------------------------------------------------------------
# Main module: Integer Invertible Watermark Network
# ---------------------------------------------------------------------------

class IntegerInvertibleWatermarkNetwork(nn.Module):
    """Normalizing-flow watermark network for pixel-domain embedding.

    Forward pass embeds a binary payload into an image via invertible
    flow blocks.  Blind extraction is handled by a separate learned
    decoder CNN that predicts the embedded bits directly.

    Parameters
    ----------
    num_blocks : int
        Number of invertible blocks (default 4).
    hidden_channels : int
        Feature channels in coupling networks (default 64).
    message_bits : int
        Number of watermark bits (default 48).
    msg_embed_dim : int
        Message embedding dimension for FiLM (default 64).
    strength : float
        Controls the magnitude of the watermark perturbation (default 0.1).
        Lower = more imperceptible but less robust.
    detection_threshold : float
        Confidence threshold for declaring watermark present (default 0.6).
    """

    def __init__(
        self,
        num_blocks: int = 4,
        hidden_channels: int = 64,
        message_bits: int = 48,
        msg_embed_dim: int = 64,
        strength: float = 0.1,
        detection_threshold: float = 0.6,
    ) -> None:
        super().__init__()
        self.message_bits = message_bits
        self.msg_embed_dim = msg_embed_dim
        self.strength = strength
        self.detection_threshold = detection_threshold

        # Message embedding MLP
        self.msg_encoder = nn.Sequential(
            nn.Linear(message_bits, msg_embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(msg_embed_dim * 2, msg_embed_dim),
        )

        # Channel padding (3 → 4 for even split in coupling layers)
        self.channel_pad = _ChannelPad()

        # Invertible blocks
        self.blocks = nn.ModuleList([
            InvertibleBlock(
                channels=4,  # 3 image + 1 pad
                hidden_channels=hidden_channels,
                msg_dim=msg_embed_dim,
            )
            for _ in range(num_blocks)
        ])

        # Blind extraction head
        self.blind_decoder = _BlindDecoder(
            message_bits=message_bits,
            base_channels=hidden_channels,
        )

    def _embed_message(self, payload: torch.Tensor) -> torch.Tensor:
        """Encode binary payload to continuous embedding."""
        return self.msg_encoder(payload.float())

    def embed(
        self,
        image: torch.Tensor,
        payload: torch.Tensor,
    ) -> torch.Tensor:
        """Embed watermark payload into image (forward flow pass).

        Parameters
        ----------
        image : (B, 3, H, W) in [0, 1]
        payload : (B, N) binary bits

        Returns
        -------
        (B, 3, H, W) watermarked image in [0, 1]
        """
        msg_emb = self._embed_message(payload)

        # Scale image to [-1, 1] for better numerical behaviour
        x = image * 2.0 - 1.0

        # Pad to 4 channels
        x = self.channel_pad(x)

        # Forward through invertible blocks
        residual = x.clone()
        for block in self.blocks:
            x = block(x, msg_emb, reverse=False)

        # Apply as residual with controlled strength
        x = residual + self.strength * (x - residual)

        # Remove padding channel and map back to [0, 1]
        x = self.channel_pad.remove(x)
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)

    def extract(
        self,
        watermarked_image: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Blind extraction of watermark payload from image.

        Uses the learned blind decoder to predict the embedded bits
        without requiring the original payload.

        Parameters
        ----------
        watermarked_image : (B, 3, H, W) in [0, 1]

        Returns
        -------
        dict with:
          - ``logits`` : (B, N) raw decoder logits
          - ``predicted_bits`` : (B, N) hard bit decisions
          - ``confidence`` : (B,) detection confidence in [0, 1]
          - ``is_watermarked`` : (B,) boolean detection decision
        """
        logits = self.blind_decoder(watermarked_image)
        predicted_bits = (logits > 0).float()

        # Confidence = mean |logit| normalized to [0, 1]
        raw_conf = logits.abs().mean(dim=1)
        confidence = torch.sigmoid(raw_conf - 0.5)

        return {
            "logits": logits,
            "predicted_bits": predicted_bits,
            "confidence": confidence,
            "is_watermarked": confidence > self.detection_threshold,
        }

    def forward(
        self,
        image: torch.Tensor,
        payload: torch.Tensor,
    ) -> torch.Tensor:
        """Embed watermark payload into image (alias for ``embed``).

        Parameters
        ----------
        image : (B, 3, H, W) in [0, 1]
        payload : (B, N) binary bits

        Returns
        -------
        (B, 3, H, W) watermarked image in [0, 1]
        """
        return self.embed(image, payload)


# ---------------------------------------------------------------------------
# Combined loss for iIWN training
# ---------------------------------------------------------------------------

class IIWNLoss(nn.Module):
    """Training loss for the Integer Invertible Watermark Network.

    L = lambda_img * MSE(watermarked, cover)       # imperceptibility
      + lambda_bit * BCE(logits, payload)           # message accuracy
      + lambda_inv * MSE(decoded_cover, cover)      # invertibility (optional)

    Parameters
    ----------
    lambda_img : float
        Weight for imperceptibility (default 1.0).
    lambda_bit : float
        Weight for bit accuracy (default 1.0).
    lambda_inv : float
        Weight for invertibility (default 10.0).
    """

    def __init__(
        self,
        lambda_img: float = 1.0,
        lambda_bit: float = 1.0,
        lambda_inv: float = 10.0,
    ) -> None:
        super().__init__()
        self.lambda_img = lambda_img
        self.lambda_bit = lambda_bit
        self.lambda_inv = lambda_inv

    def forward(
        self,
        cover: torch.Tensor,
        watermarked: torch.Tensor,
        logits: torch.Tensor | None = None,
        payload: torch.Tensor | None = None,
        recovered_cover: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Parameters
        ----------
        cover : (B, 3, H, W) — original image
        watermarked : (B, 3, H, W) — iIWN output
        logits : (B, N) or None — blind decoder logits
        payload : (B, N) or None — ground-truth bits for BCE
        recovered_cover : (B, 3, H, W) or None — inverse pass output

        Returns
        -------
        dict with ``loss``, ``img_mse``, ``psnr``, and optionally
        ``bit_bce``, ``bit_acc``, ``inv_mse``.
        """
        img_mse = F.mse_loss(watermarked, cover)
        total = self.lambda_img * img_mse

        result: dict[str, torch.Tensor] = {
            "img_mse": img_mse.detach(),
        }

        # Bit accuracy loss
        if logits is not None and payload is not None:
            bit_bce = F.binary_cross_entropy_with_logits(logits, payload.float())
            total = total + self.lambda_bit * bit_bce
            result["bit_bce"] = bit_bce.detach()
            with torch.no_grad():
                pred_bits = (logits > 0).float()
                result["bit_acc"] = (pred_bits == payload.float()).float().mean()

        # Invertibility loss
        if recovered_cover is not None:
            inv_mse = F.mse_loss(recovered_cover, cover)
            total = total + self.lambda_inv * inv_mse
            result["inv_mse"] = inv_mse.detach()

        # PSNR (from MSE)
        with torch.no_grad():
            psnr = -10.0 * torch.log10(img_mse.clamp(min=1e-10))

        result["loss"] = total
        result["psnr"] = psnr

        return result
