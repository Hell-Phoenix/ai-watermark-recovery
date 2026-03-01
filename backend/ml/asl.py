"""Attack Simulation Layer (ASL).

A differentiable attack pipeline inserted between the encoder and decoder
during training.  It randomly applies one or more attacks from a
configurable menu so the watermark system learns to survive:

  1. **JPEG compression** — differentiable proxy (``DifferentiableJPEG``)
  2. **Random crop** — STN-based (``DifferentiableCrop``)
  3. **Geometric distortion** — rotation / scale / shear (``GeometricDistortion``)
  4. **Gaussian noise** — additive i.i.d.
  5. **Gaussian blur** — random kernel size & sigma
  6. **Diffusion regeneration** — forward DDPM noise + optional partial denoise
  7. **Identity** — clean pass-through (keeps a learning signal for easy cases)

The layer's ``severity`` attribute (0 → 1) is driven by the curriculum
scheduler; it interpolates every attack parameter between its *easy* and
*hard* extremes.

Reference
---------
  Fernandez et al., "The Stable Signature", ICCV 2023
  Zhu et al., "HiDDeN", ECCV 2018
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.ml.jsnet import DifferentiableJPEG
from backend.ml.stn_crop import DifferentiableCrop, GeometricDistortion

# ────────────────────────────────────────────────────────────
# Attack-parameter ranges (easy → hard, interpolated by severity)
# ────────────────────────────────────────────────────────────


@dataclass
class AttackConfig:
    """Specifies the parameter ranges for each attack.

    ``severity`` ∈ [0, 1] linearly interpolates between the *_easy* and
    *_hard* values.  The curriculum scheduler simply updates ``severity``.
    """

    # JPEG quality factor
    jpeg_qf_easy: float = 70.0
    jpeg_qf_hard: float = 5.0

    # Crop ratio (area fraction retained)
    crop_ratio_easy: float = 0.80
    crop_ratio_hard: float = 0.05

    # Gaussian noise std-dev
    noise_std_easy: float = 0.01
    noise_std_hard: float = 0.10

    # Gaussian blur sigma
    blur_sigma_easy: float = 0.5
    blur_sigma_hard: float = 3.0
    blur_kernel_size: int = 7  # fixed kernel size, sigma controls strength

    # Geometric distortion
    geo_angle_easy: float = 5.0  # max degrees
    geo_angle_hard: float = 45.0
    geo_scale_easy: tuple[float, float] = (0.95, 1.05)
    geo_scale_hard: tuple[float, float] = (0.6, 1.4)
    geo_shear_easy: float = 0.05
    geo_shear_hard: float = 0.3

    # Diffusion regeneration attack
    diffusion_timestep_easy: int = 50
    diffusion_timestep_hard: int = 900
    diffusion_beta_start: float = 1e-4
    diffusion_beta_end: float = 0.02
    diffusion_num_steps: int = 1000

    # Per-attack application probability (can be overridden by curriculum)
    attack_probs: dict[str, float] = field(
        default_factory=lambda: {
            "jpeg": 0.3,
            "crop": 0.3,
            "noise": 0.2,
            "blur": 0.15,
            "geometric": 0.15,
            "diffusion": 0.10,
            "identity": 0.15,
        }
    )


def _lerp(easy: float, hard: float, severity: float) -> float:
    """Linear interpolation: severity=0 → easy, severity=1 → hard."""
    return easy + (hard - easy) * severity


def _lerp_tuple(
    easy: tuple[float, float],
    hard: tuple[float, float],
    severity: float,
) -> tuple[float, float]:
    return (_lerp(easy[0], hard[0], severity), _lerp(easy[1], hard[1], severity))


# ────────────────────────────────────────────────────────────
# Individual differentiable attack modules
# ────────────────────────────────────────────────────────────


class GaussianNoise(nn.Module):
    """Additive Gaussian noise with configurable std-dev."""

    def __init__(self, std: float = 0.03) -> None:
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.randn_like(x) * self.std
            return (x + noise).clamp(0.0, 1.0)
        return x


class GaussianBlur(nn.Module):
    """Differentiable Gaussian blur (depthwise conv)."""

    def __init__(self, kernel_size: int = 7, sigma: float = 1.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def _make_kernel(self, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ks = self.kernel_size
        ax = torch.arange(ks, device=device, dtype=dtype) - ks // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * self.sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.expand(channels, 1, ks, ks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pad = self.kernel_size // 2
        kernel = self._make_kernel(C, x.device, x.dtype)
        return F.conv2d(x, kernel, padding=pad, groups=C).clamp(0.0, 1.0)


class DiffusionNoiseAttack(nn.Module):
    """Simulates a diffusion regeneration attack (D2RA / DAWN).

    Applies the *forward* DDPM noise process at a random timestep to
    degrade the watermark signal, mimicking the first half of a
    regeneration attack.

    The noise schedule follows a linear beta schedule.
    """

    def __init__(
        self,
        max_timestep: int = 400,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        num_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.max_timestep = max_timestep
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bar", alpha_bar)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward diffusion noise at a random timestep.

        Parameters
        ----------
        x : Tensor  (B, C, H, W) in [0, 1]

        Returns
        -------
        Tensor  (B, C, H, W) — noised image, clamped to [0, 1]
        """
        if not self.training:
            return x

        B = x.shape[0]
        # Sample a random timestep per image (1 … max_timestep)
        t = torch.randint(1, self.max_timestep + 1, (B,), device=x.device)

        # Gather cumulative alpha for each sample
        a_bar = self.alpha_bar[t].view(B, 1, 1, 1)  # (B, 1, 1, 1)

        # Forward diffusion:  x_t = sqrt(ā_t) * x_0  + sqrt(1 - ā_t) * ε
        noise = torch.randn_like(x)
        x_noised = a_bar.sqrt() * x + (1.0 - a_bar).sqrt() * noise
        return x_noised.clamp(0.0, 1.0)


# ────────────────────────────────────────────────────────────
# Main ASL module
# ────────────────────────────────────────────────────────────


class AttackSimulationLayer(nn.Module):
    """Orchestrates all differentiable attack types.

    Usage
    -----
    >>> asl = AttackSimulationLayer()
    >>> asl.severity = 0.0   # easy (beginning of training)
    >>> attacked = asl(watermarked_images)
    >>> asl.severity = 0.8   # hard (late training, set by curriculum)
    >>> attacked = asl(watermarked_images)

    The layer randomly selects attack(s) per forward pass based on the
    probabilities defined in ``AttackConfig.attack_probs``.

    Parameters
    ----------
    config : AttackConfig
        Attack parameter ranges.
    num_attacks : int
        Number of attacks to apply per sample (sampled randomly each call).
        Set to 1 for single-attack training.
    """

    def __init__(
        self,
        config: AttackConfig | None = None,
        num_attacks: int = 1,
    ) -> None:
        super().__init__()
        self.config = config or AttackConfig()
        self.num_attacks = num_attacks
        self._severity: float = 0.0

        # Instantiate sub-modules (parameters will be updated each forward)
        self.jpeg = DifferentiableJPEG(quality=self.config.jpeg_qf_easy)
        self.crop = DifferentiableCrop(
            min_ratio=self.config.crop_ratio_hard,
            max_ratio=self.config.crop_ratio_easy,
        )
        self.geo = GeometricDistortion()
        self.noise = GaussianNoise()
        self.blur = GaussianBlur(kernel_size=self.config.blur_kernel_size)
        self.diffusion = DiffusionNoiseAttack(
            max_timestep=self.config.diffusion_timestep_easy,
            beta_start=self.config.diffusion_beta_start,
            beta_end=self.config.diffusion_beta_end,
            num_steps=self.config.diffusion_num_steps,
        )

    # --- severity property (updated by curriculum scheduler) ---

    @property
    def severity(self) -> float:
        """Current attack severity ∈ [0, 1]."""
        return self._severity

    @severity.setter
    def severity(self, value: float) -> None:
        self._severity = max(0.0, min(1.0, value))
        self._update_attack_params()

    def _update_attack_params(self) -> None:
        """Recompute all attack parameters from the current severity."""
        s = self._severity
        cfg = self.config

        # JPEG
        self.jpeg.default_quality = _lerp(cfg.jpeg_qf_easy, cfg.jpeg_qf_hard, s)

        # Crop
        min_r = _lerp(cfg.crop_ratio_easy, cfg.crop_ratio_hard, s)
        self.crop.min_ratio = min_r
        self.crop.max_ratio = max(min_r + 0.05, _lerp(1.0, cfg.crop_ratio_easy, s))

        # Noise
        self.noise.std = _lerp(cfg.noise_std_easy, cfg.noise_std_hard, s)

        # Blur
        self.blur.sigma = _lerp(cfg.blur_sigma_easy, cfg.blur_sigma_hard, s)

        # Geometric
        self.geo.max_angle = _lerp(cfg.geo_angle_easy, cfg.geo_angle_hard, s)
        self.geo.scale_range = _lerp_tuple(cfg.geo_scale_easy, cfg.geo_scale_hard, s)
        self.geo.max_shear = _lerp(cfg.geo_shear_easy, cfg.geo_shear_hard, s)

        # Diffusion
        self.diffusion.max_timestep = int(
            _lerp(cfg.diffusion_timestep_easy, cfg.diffusion_timestep_hard, s)
        )

    # --- attack dispatch ---

    _ATTACK_NAMES: list[str] = [
        "jpeg",
        "crop",
        "noise",
        "blur",
        "geometric",
        "diffusion",
        "identity",
    ]

    def _pick_attacks(self) -> list[str]:
        """Sample ``num_attacks`` attack names according to their probabilities."""
        probs = self.config.attack_probs
        names = [n for n in self._ATTACK_NAMES if n in probs]
        weights = [probs[n] for n in names]
        chosen = random.choices(names, weights=weights, k=self.num_attacks)
        return chosen

    def _apply_single(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if name == "jpeg":
            return self.jpeg(x)
        if name == "crop":
            return self.crop(x)
        if name == "noise":
            return self.noise(x)
        if name == "blur":
            return self.blur(x)
        if name == "geometric":
            return self.geo(x)
        if name == "diffusion":
            return self.diffusion(x)
        # identity
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random attack(s) to the watermarked images.

        Parameters
        ----------
        x : Tensor  (B, 3, H, W) in [0, 1]

        Returns
        -------
        Tensor  (B, 3, H, W) — attacked images in [0, 1]
        """
        if not self.training:
            return x  # no attacks at inference time

        attacks = self._pick_attacks()
        for name in attacks:
            x = self._apply_single(x, name)
        return x.clamp(0.0, 1.0)

    def extra_repr(self) -> str:
        return f"severity={self._severity:.2f}, num_attacks={self.num_attacks}"
