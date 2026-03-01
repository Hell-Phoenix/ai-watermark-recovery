"""DDIM Inversion for latent watermark extraction.

DDIM inversion reverses the deterministic DDIM sampling process to
reconstruct the **initial noise latent** z_T from a given image x_0.
This is critical for WIND-style latent watermarking where the watermark
is encoded as a pattern in the initial noise (``z_w = z + α·P(H(salt‖id))``).

The key insight: DDIM sampling is (approximately) reversible because it is
deterministic (η=0).  By running the process backwards — from x_0 towards
x_T — we can recover z_T and check whether it contains a watermark pattern.

This module supports:
  1. **Exact inversion** (when the diffusion model is available) — runs
     DDIM forward (inversion) steps using the model's noise predictions.
  2. **Approximate inversion** (blind mode) — uses a lightweight learned
     encoder to directly predict z_T from x_0 without access to the
     original diffusion model.  Trained via MSE against exact inversions.

References:
  - DDIM: Song et al., ICLR 2021 — Eq. 12 (deterministic sampling)
  - WIND: Wei et al., ICLR 2025 — initial noise watermarking
  - Null-text inversion: Mokady et al., CVPR 2023
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Noise schedule helpers (shared with diffusion_restore.py)
# ---------------------------------------------------------------------------

def _linear_beta_schedule(
    num_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps)


def _cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    f = torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0, 0.999).float()


# ---------------------------------------------------------------------------
# DDIM Inversion (exact — requires noise prediction model)
# ---------------------------------------------------------------------------

class DDIMInversion(nn.Module):
    """DDIM Inversion: x_0 → z_T recovery.

    Given an image x_0 and a noise prediction model ε_θ, reconstructs
    the initial noise z_T by running the deterministic DDIM process
    in reverse (from t=0 towards t=T).

    The inversion formula (DDIM Eq. 12, reversed):
        x_{t+1} = √(ᾱ_{t+1}) · x̂_0(x_t) + √(1 - ᾱ_{t+1}) · ε_θ(x_t, t)

    Parameters
    ----------
    noise_predictor : nn.Module
        Model that predicts noise ε given (x_t, t). Can be a diffusion
        U-Net or any compatible model.
    num_steps : int
        Total diffusion timesteps T (default 1000).
    schedule : str
        Noise schedule type (default "linear").
    inversion_steps : int
        Number of steps for inversion (default 50). More steps = more accurate.
    """

    def __init__(
        self,
        noise_predictor: nn.Module,
        num_steps: int = 1000,
        schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        inversion_steps: int = 50,
    ) -> None:
        super().__init__()
        self.noise_predictor = noise_predictor
        self.num_steps = num_steps
        self.inversion_steps = inversion_steps

        # Build schedule
        if schedule == "cosine":
            betas = _cosine_beta_schedule(num_steps)
        else:
            betas = _linear_beta_schedule(num_steps, beta_start, beta_end)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # Freeze noise predictor
        for param in self.noise_predictor.parameters():
            param.requires_grad = False

    def _get_timestep_subsequence(self, steps: int) -> list[int]:
        """Create evenly spaced timestep subsequence for inversion."""
        return torch.linspace(0, self.num_steps - 1, steps, dtype=torch.long).tolist()

    @torch.no_grad()
    def invert(
        self,
        x_0: torch.Tensor,
        num_steps: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run DDIM inversion: x_0 → z_T.

        Parameters
        ----------
        x_0 : (B, C, H, W) — input image
        num_steps : int or None — override inversion steps

        Returns
        -------
        z_T : (B, C, H, W) — reconstructed initial noise
        intermediates : list of tensors — all intermediate x_t values
        """
        steps = num_steps or self.inversion_steps
        timesteps = self._get_timestep_subsequence(steps)

        x_t = x_0
        intermediates: list[torch.Tensor] = [x_t]

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # Current and next alpha_bar
            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_next = self.alphas_cumprod[t_next]

            B = x_t.shape[0]
            t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)

            # Predict noise at current timestep
            noise_pred = self.noise_predictor(x_t, t_tensor)

            # Predict x_0 from x_t
            x0_pred = (x_t - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()

            # DDIM forward step (inversion): x_t → x_{t_next}
            x_t = (
                alpha_bar_next.sqrt() * x0_pred
                + (1 - alpha_bar_next).sqrt() * noise_pred
            )

            intermediates.append(x_t)

        return x_t, intermediates

    def forward(
        self, x_0: torch.Tensor, num_steps: int | None = None
    ) -> torch.Tensor:
        """Invert x_0 to z_T."""
        z_T, _ = self.invert(x_0, num_steps)
        return z_T


# ---------------------------------------------------------------------------
# Approximate (Blind) DDIM Inversion
# ---------------------------------------------------------------------------

class _InversionEncoder(nn.Module):
    """Lightweight encoder that directly predicts z_T from x_0.

    Used for blind extraction when the original diffusion model is
    not available.  Trained via MSE against exact DDIM inversions.

    Architecture: ResNet-style encoder → spatial feature map (same size as
    input but representing the noise latent).
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        ch = base_channels

        self.encoder = nn.Sequential(
            # Initial conv
            nn.Conv2d(in_channels, ch, 7, padding=3, bias=False),
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(inplace=True),

            # Residual blocks at full resolution
            _ResBlock(ch, ch),
            _ResBlock(ch, ch),

            # Down to 1/2
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, ch * 2), ch * 2),
            nn.SiLU(inplace=True),
            _ResBlock(ch * 2, ch * 2),
            _ResBlock(ch * 2, ch * 2),

            # Down to 1/4
            nn.Conv2d(ch * 2, ch * 4, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, ch * 4), ch * 4),
            nn.SiLU(inplace=True),
            _ResBlock(ch * 4, ch * 4),
            _ResBlock(ch * 4, ch * 4),
        )

        self.decoder = nn.Sequential(
            # Up to 1/2
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, ch * 2), ch * 2),
            nn.SiLU(inplace=True),
            _ResBlock(ch * 2, ch * 2),

            # Up to full
            nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(inplace=True),
            _ResBlock(ch, ch),

            # Output noise prediction
            nn.Conv2d(ch, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.decoder(features)


class _ResBlock(nn.Module):
    """Simple residual block for the inversion encoder."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.skip(x) + self.block(x))


class BlindDDIMInversion(nn.Module):
    """Approximate DDIM inversion without access to the diffusion model.

    A learned encoder directly maps x_0 → z_T, enabling watermark
    extraction from images generated by unknown diffusion models.

    This is trained by:
      1. Running exact DDIM inversion on training images to get z_T targets
      2. Training the encoder via MSE: ||encoder(x_0) - z_T||²

    Parameters
    ----------
    in_channels : int
        Image channels (default 3).
    base_channels : int
        Feature channels in the encoder (default 64).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = _InversionEncoder(in_channels, base_channels)

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        """Predict initial noise z_T from image x_0.

        Parameters
        ----------
        x_0 : (B, C, H, W) — input image in [0, 1]

        Returns
        -------
        z_T : (B, C, H, W) — predicted initial noise latent
        """
        return self.encoder(x_0)

    def training_loss(
        self,
        x_0: torch.Tensor,
        z_T_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute MSE training loss against exact inversion target.

        Parameters
        ----------
        x_0 : (B, C, H, W) — input image
        z_T_target : (B, C, H, W) — ground truth z_T from exact inversion

        Returns
        -------
        dict with ``loss`` (MSE), ``cosine_sim`` (diagnostic).
        """
        z_T_pred = self.encoder(x_0)
        loss = F.mse_loss(z_T_pred, z_T_target)

        with torch.no_grad():
            # Cosine similarity as a diagnostic metric
            pred_flat = z_T_pred.flatten(1)
            target_flat = z_T_target.flatten(1)
            cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean()

        return {"loss": loss, "cosine_sim": cos_sim}


# ---------------------------------------------------------------------------
# Watermark pattern extractor (for WIND-style latent watermarks)
# ---------------------------------------------------------------------------

class LatentWatermarkExtractor(nn.Module):
    """Extract watermark pattern from recovered noise latent.

    Given z_T (from DDIM inversion), checks whether it contains the
    WIND-style Fourier pattern P(H(salt‖id)) by computing correlation
    with the expected pattern.

    Parameters
    ----------
    pattern_dim : int
        Dimension of the watermark pattern (default 48).
    latent_channels : int
        Number of channels in the noise latent (default 3 for RGB,
        4 for SD latent space).
    """

    def __init__(
        self,
        pattern_dim: int = 48,
        latent_channels: int = 3,
    ) -> None:
        super().__init__()
        self.pattern_dim = pattern_dim

        # Small network to extract pattern bits from noise latent
        self.extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),  # (B, C, 8, 8)
            nn.Flatten(),
            nn.Linear(latent_channels * 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, pattern_dim),  # logits
        )

    def forward(self, z_T: torch.Tensor) -> torch.Tensor:
        """Extract watermark bits from noise latent.

        Parameters
        ----------
        z_T : (B, C, H, W) — noise latent from DDIM inversion

        Returns
        -------
        (B, pattern_dim) — bit logits
        """
        return self.extractor(z_T)

    def detect(
        self,
        z_T: torch.Tensor,
        expected_bits: torch.Tensor,
        threshold: float = 0.7,
    ) -> dict[str, torch.Tensor]:
        """Detect watermark presence and compare to expected bits.

        Parameters
        ----------
        z_T : (B, C, H, W) — noise latent
        expected_bits : (B, pattern_dim) — expected bit pattern
        threshold : float — minimum bit accuracy for detection

        Returns
        -------
        dict with ``logits``, ``predicted_bits``, ``bit_accuracy``,
        ``is_watermarked`` (bool per sample).
        """
        logits = self.forward(z_T)
        predicted_bits = (logits > 0).float()

        bit_accuracy = (predicted_bits == expected_bits.float()).float().mean(dim=1)
        is_watermarked = bit_accuracy > threshold

        return {
            "logits": logits,
            "predicted_bits": predicted_bits,
            "bit_accuracy": bit_accuracy,
            "is_watermarked": is_watermarked,
        }
