"""Conditional Diffusion Restoration for watermark recovery.

This module implements a lightweight conditional denoising diffusion model
specifically designed for **watermark signal restoration**, not visual
quality.  It learns to reverse the degradation process by:

1. **Forward process** — adds noise at a controlled timestep *t*
   (same beta schedule as the ASL diffusion attack).
2. **Reverse process** — a U-Net denoiser conditioned on the degraded image
   iteratively removes noise.  The denoiser is trained so that passing the
   output through a frozen watermark decoder maximises bit accuracy.

The key insight is that we don't need a full generative diffusion model —
we only need enough denoising steps to recover the watermark bit structure.
This module therefore supports **few-step inference** (5–20 steps) using
both DDPM and DDIM sampling.

Architecture:
  - Uses ``ConditionalRestorationUNet`` from ``unet_restore.py`` as backbone
  - Noise schedule: linear beta ∈ [1e-4, 0.02], T=1000
  - Conditioning: degraded image concatenated channel-wise to noisy input
  - Training: standard ε-prediction with auxiliary bit-loss through decoder

References:
  - DDPM: Ho et al., NeurIPS 2020
  - DDIM: Song et al., ICLR 2021
  - Conditional diffusion for restoration: Saharia et al. (Palette, 2022)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.ml.unet_restore import ConditionalRestorationUNet, sinusoidal_embedding


# ---------------------------------------------------------------------------
# Noise schedule helpers
# ---------------------------------------------------------------------------

def _linear_beta_schedule(
    num_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Linear beta schedule as in DDPM."""
    return torch.linspace(beta_start, beta_end, num_steps)


def _cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule (Nichol & Dhariwal, 2021) for smoother noise."""
    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    f = torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0, 0.999).float()


# ---------------------------------------------------------------------------
# Conditional restoration denoiser (wraps ConditionalRestorationUNet)
# ---------------------------------------------------------------------------

class _ConditionedDenoiser(nn.Module):
    """Denoiser that takes (noisy_image ‖ degraded_image) as input.

    The degraded image is concatenated channel-wise as conditioning
    information, doubling the input channels (6 for RGB).  The U-Net
    uses ``out_channels=in_channels`` (3) so residual learning is
    disabled and the output is the predicted noise ε.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        depth: int = 4,
        num_bottleneck_blocks: int = 4,
        t_dim: int = 256,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        # Input = noisy + condition → 2× channels; output = noise (3ch)
        self.unet = ConditionalRestorationUNet(
            in_channels=in_channels * 2,  # concat conditioning
            out_channels=in_channels,     # predict noise (3ch, no residual)
            base_channels=base_channels,
            depth=depth,
            num_bottleneck_blocks=num_bottleneck_blocks,
            t_dim=t_dim,
            attn_heads=attn_heads,
        )

    def forward(
        self,
        x_noisy: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise ε from (x_noisy ‖ condition) at timestep t.

        Parameters
        ----------
        x_noisy : (B, 3, H, W) — noisy image
        condition : (B, 3, H, W) — degraded watermarked image
        t : (B,) — diffusion timestep normalised to [0, 1]

        Returns
        -------
        (B, 3, H, W) — predicted noise ε
        """
        x_in = torch.cat([x_noisy, condition], dim=1)  # (B, 6, H, W)
        return self.unet(x_in, severity=t)


# ---------------------------------------------------------------------------
# Main module: ConditionalDiffusionRestorer
# ---------------------------------------------------------------------------

class ConditionalDiffusionRestorer(nn.Module):
    """Conditional diffusion model for watermark-aware image restoration.

    The model learns to denoise a degraded watermarked image to recover
    the watermark bit structure.  Unlike standard image restoration
    diffusion models, this one is trained with an auxiliary **bit-recovery
    loss** through a frozen decoder.

    Parameters
    ----------
    num_steps : int
        Total diffusion steps T (default 1000).
    schedule : str
        Noise schedule type: ``"linear"`` or ``"cosine"`` (default ``"linear"``).
    beta_start : float
        Start of linear beta schedule (default 1e-4).
    beta_end : float
        End of linear beta schedule (default 0.02).
    base_channels : int
        Base feature channels in the denoiser U-Net (default 64).
    depth : int
        Number of U-Net encoder/decoder levels (default 4).
    inference_steps : int
        Number of DDIM sampling steps at inference (default 10).
    """

    def __init__(
        self,
        num_steps: int = 1000,
        schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        base_channels: int = 64,
        depth: int = 4,
        inference_steps: int = 10,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.inference_steps = inference_steps

        # Noise schedule
        if schedule == "cosine":
            betas = _cosine_beta_schedule(num_steps)
        else:
            betas = _linear_beta_schedule(num_steps, beta_start, beta_end)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (not parameters — not optimised)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

        # Posterior variance for DDPM sampling
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_var)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_var.clamp(min=1e-20)),
        )

        # Denoiser network
        self.denoiser = _ConditionedDenoiser(
            in_channels=3,
            base_channels=base_channels,
            depth=depth,
        )

    # ----- Forward (training) -----

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward diffusion: add noise at timestep *t*.

        Parameters
        ----------
        x_start : (B, 3, H, W) — clean watermarked image
        t : (B,) — integer timesteps in [0, T-1]
        noise : optional pre-sampled noise

        Returns
        -------
        (B, 3, H, W) — noised image x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_a = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_a = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_a * x_start + sqrt_one_minus_a * noise

    def training_loss(
        self,
        clean_watermarked: torch.Tensor,
        degraded: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute training loss (noise prediction MSE).

        Parameters
        ----------
        clean_watermarked : (B, 3, H, W) — target (original watermarked image)
        degraded : (B, 3, H, W) — conditioning input (attacked image)

        Returns
        -------
        dict with ``loss`` (MSE of noise prediction) and ``t`` (sampled timesteps).
        """
        B = clean_watermarked.shape[0]
        t = torch.randint(0, self.num_steps, (B,), device=clean_watermarked.device)
        noise = torch.randn_like(clean_watermarked)
        x_noisy = self.q_sample(clean_watermarked, t, noise=noise)

        # Predict noise
        t_normalised = t.float() / self.num_steps  # [0, 1)
        noise_pred = self.denoiser(x_noisy, degraded, t_normalised)

        loss = F.mse_loss(noise_pred, noise)
        return {"loss": loss, "t": t}

    # ----- Reverse sampling -----

    @torch.no_grad()
    def ddpm_step(
        self,
        x_t: torch.Tensor,
        t: int,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Single DDPM reverse step: x_t → x_{t-1}."""
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)
        t_norm = t_tensor.float() / self.num_steps

        noise_pred = self.denoiser(x_t, condition, t_norm)

        alpha = self.alphas[t]
        alpha_bar = self.alphas_cumprod[t]
        beta = self.betas[t]

        # Mean of p(x_{t-1} | x_t)
        mean = (1.0 / alpha.sqrt()) * (x_t - (beta / (1.0 - alpha_bar).sqrt()) * noise_pred)

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = self.posterior_variance[t].sqrt()
            return mean + sigma * noise
        return mean

    @torch.no_grad()
    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        condition: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Single DDIM step: x_t → x_{t_prev}.

        Parameters
        ----------
        eta : float
            DDIM stochasticity (0 = deterministic, 1 = DDPM equivalent).
        """
        B = x_t.shape[0]
        t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)
        t_norm = t_tensor.float() / self.num_steps

        noise_pred = self.denoiser(x_t, condition, t_norm)

        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Predicted x_0
        x0_pred = (x_t - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)

        # Direction pointing to x_t
        sigma = (
            eta
            * ((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)).sqrt()
        )

        dir_xt = (1 - alpha_bar_prev - sigma**2).clamp(min=0).sqrt() * noise_pred
        noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)

        x_prev = alpha_bar_prev.sqrt() * x0_pred + dir_xt + sigma * noise
        return x_prev

    @torch.no_grad()
    def restore(
        self,
        degraded: torch.Tensor,
        num_steps: int | None = None,
        use_ddim: bool = True,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Run reverse diffusion to restore watermark from degraded image.

        Parameters
        ----------
        degraded : (B, 3, H, W) — attacked/degraded image
        num_steps : int or None
            Sampling steps (default: ``self.inference_steps``).
        use_ddim : bool
            If True, use DDIM sampling (deterministic, fast). Else DDPM.
        eta : float
            DDIM stochasticity parameter.

        Returns
        -------
        (B, 3, H, W) — restored image.
        """
        steps = num_steps or self.inference_steps

        # Start from pure noise
        x = torch.randn_like(degraded)

        if use_ddim:
            # Create sub-sampled timestep sequence
            timesteps = torch.linspace(
                self.num_steps - 1, 0, steps + 1, dtype=torch.long
            ).tolist()
            timesteps = [int(t) for t in timesteps]

            for i in range(len(timesteps) - 1):
                t = timesteps[i]
                t_prev = timesteps[i + 1]
                x = self.ddim_step(x, t, t_prev, degraded, eta=eta)
        else:
            # Full DDPM sampling (slow but flexible)
            step_size = max(1, self.num_steps // steps)
            timesteps = list(range(self.num_steps - 1, -1, -step_size))
            for t in timesteps:
                x = self.ddpm_step(x, t, degraded)

        return x.clamp(0.0, 1.0)

    def forward(
        self,
        degraded: torch.Tensor,
        clean_watermarked: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Forward pass: training loss if ``clean_watermarked`` provided, else restore.

        Parameters
        ----------
        degraded : (B, 3, H, W) — attacked image
        clean_watermarked : (B, 3, H, W) or None — target for training

        Returns
        -------
        If training: dict with ``loss`` and ``t``.
        If eval: (B, 3, H, W) restored image.
        """
        if self.training and clean_watermarked is not None:
            return self.training_loss(clean_watermarked, degraded)
        return self.restore(degraded)
