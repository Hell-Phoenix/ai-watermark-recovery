"""Differentiable spatial transforms — crop, rotation, scale, shear.

All transforms use ``torch.nn.functional.grid_sample`` so that gradients
flow back through the spatial transformation into the encoder.

Key modules
-----------
``DifferentiableCrop``
    Simulates random cropping (5 %–100 % canvas) with gradient-preserving
    bilinear interpolation via Spatial Transformer Networks (STN).

``GeometricDistortion``
    Applies random rotation / scale / shear using a learnable or
    stochastic affine matrix, again backed by ``grid_sample``.

``SpatialAttackCompose``
    Convenience wrapper that randomly selects crop and/or geometric
    distortion per sample in a batch.

Reference
---------
  - Jaderberg et al., "Spatial Transformer Networks", NeurIPS 2015
  - Zhu et al., "HiDDeN", ECCV 2018 (random crop attack)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────
# Differentiable crop via affine grid_sample
# ────────────────────────────────────────────────────────────


class DifferentiableCrop(nn.Module):
    """Gradient-preserving random crop.

    During the forward pass a random crop region is selected (parameterised
    by centre coordinates and a crop ratio).  The crop is obtained via
    bilinear ``grid_sample`` which is fully differentiable.

    Parameters
    ----------
    min_ratio : float
        Minimum crop area as fraction of the full image (default 0.05 = 5 %).
    max_ratio : float
        Maximum crop area fraction (default 1.0 = no crop).
    output_size : tuple[int, int] | None
        If given, the cropped region is resized to this (H, W).
        If *None*, the output keeps the original spatial dimensions
        (the crop is embedded in zero-padding).
    """

    def __init__(
        self,
        min_ratio: float = 0.05,
        max_ratio: float = 1.0,
        output_size: Tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a random differentiable crop.

        Parameters
        ----------
        x : Tensor  (B, C, H, W)
            Input images in [0, 1].

        Returns
        -------
        Tensor  (B, C, H_out, W_out)
            Cropped (and optionally resized) images.
        """
        B, C, H, W = x.shape

        # --- sample crop parameters per image in the batch ---
        # Each image gets an independent random crop ratio & position.
        ratio = torch.empty(B, device=x.device).uniform_(self.min_ratio, self.max_ratio)
        side = ratio.sqrt()  # square crop: side_fraction = sqrt(area_fraction)

        # Random centre within valid range so the crop stays inside the image
        half = side / 2.0
        cx = torch.empty(B, device=x.device).uniform_(0.0, 1.0) * (1.0 - side) + half
        cy = torch.empty(B, device=x.device).uniform_(0.0, 1.0) * (1.0 - side) + half

        # --- build affine matrices ---
        # grid_sample expects coords in [−1, 1].  We need an affine that maps
        # the output grid to the crop window in the input.
        #   mapping:  output_coord ∈ [−1, 1]  →  input_coord = centre + output_coord * half_side
        #   in matrix form:  [sx  0  tx]
        #                    [0  sy  ty]
        sx = side  # scale x
        sy = side  # scale y
        tx = 2.0 * cx - 1.0  # translate x  (convert [0,1] centre → [−1,1])
        ty = 2.0 * cy - 1.0  # translate y

        theta = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
        theta[:, 0, 0] = sx
        theta[:, 1, 1] = sy
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        out_h = self.output_size[0] if self.output_size else H
        out_w = self.output_size[1] if self.output_size else W

        grid = F.affine_grid(theta, (B, C, out_h, out_w), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


# ────────────────────────────────────────────────────────────
# Differentiable geometric distortion (rotation / scale / shear)
# ────────────────────────────────────────────────────────────


class GeometricDistortion(nn.Module):
    """Random affine distortion — rotation, scale, shear.

    Each forward pass samples a random transformation.  All ops are
    implemented via a single ``grid_sample`` call.

    Parameters
    ----------
    max_angle : float
        Maximum rotation in degrees (sampled uniformly ±max_angle).
    scale_range : tuple[float, float]
        (min_scale, max_scale).  1.0 = identity.
    max_shear : float
        Maximum shear in radians (sampled uniformly ±max_shear).
    """

    def __init__(
        self,
        max_angle: float = 30.0,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        max_shear: float = 0.2,
    ) -> None:
        super().__init__()
        self.max_angle = max_angle
        self.scale_range = scale_range
        self.max_shear = max_shear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random geometric distortion.

        Parameters
        ----------
        x : Tensor  (B, C, H, W)

        Returns
        -------
        Tensor  (B, C, H, W)
        """
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # --- per-sample random parameters ---
        angles = torch.empty(B, device=device).uniform_(
            -self.max_angle, self.max_angle
        )  # degrees
        scales = torch.empty(B, device=device).uniform_(*self.scale_range)
        shear_x = torch.empty(B, device=device).uniform_(-self.max_shear, self.max_shear)
        shear_y = torch.empty(B, device=device).uniform_(-self.max_shear, self.max_shear)

        # --- build 2×3 affine matrix per sample ---
        rad = angles * (math.pi / 180.0)
        cos_a = torch.cos(rad)
        sin_a = torch.sin(rad)

        # Rotation matrix  R
        # Scale             S = diag(s, s)
        # Shear             Sh = [[1, shx], [shy, 1]]
        # Combined          M = S @ R @ Sh  (no translation, centred at origin)

        # Row 0: [s*(cos*1 + sin*shy),  s*(cos*shx + sin*1),  0]
        # Row 1: [s*(-sin*1 + cos*shy), s*(-sin*shx + cos*1), 0]

        m00 = scales * (cos_a + sin_a * shear_y)
        m01 = scales * (cos_a * shear_x + sin_a)
        m10 = scales * (-sin_a + cos_a * shear_y)
        m11 = scales * (-sin_a * shear_x + cos_a)

        theta = torch.zeros(B, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = m00
        theta[:, 0, 1] = m01
        theta[:, 1, 0] = m10
        theta[:, 1, 1] = m11
        # translation stays 0 → centred distortion

        grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


# ────────────────────────────────────────────────────────────
# Combined spatial attack
# ────────────────────────────────────────────────────────────


class SpatialAttackCompose(nn.Module):
    """Randomly applies crop and/or geometric distortion.

    With probability ``p_crop`` a differentiable crop is applied first,
    then with probability ``p_geo`` a geometric distortion is applied.

    Parameters
    ----------
    crop_kwargs : dict
        Keyword arguments forwarded to ``DifferentiableCrop``.
    geo_kwargs : dict
        Keyword arguments forwarded to ``GeometricDistortion``.
    p_crop : float
        Probability of applying the crop (default 0.5).
    p_geo : float
        Probability of applying geometric distortion (default 0.3).
    """

    def __init__(
        self,
        crop_kwargs: dict | None = None,
        geo_kwargs: dict | None = None,
        p_crop: float = 0.5,
        p_geo: float = 0.3,
    ) -> None:
        super().__init__()
        self.crop = DifferentiableCrop(**(crop_kwargs or {}))
        self.geo = GeometricDistortion(**(geo_kwargs or {}))
        self.p_crop = p_crop
        self.p_geo = p_geo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and torch.rand(1).item() < self.p_crop:
            x = self.crop(x)
        if self.training and torch.rand(1).item() < self.p_geo:
            x = self.geo(x)
        return x
