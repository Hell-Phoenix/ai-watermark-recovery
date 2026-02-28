"""Differentiable JPEG compression approximation (JSNet).

Implements a fully differentiable proxy for JPEG compression that can be
used inside a training loop.  Real JPEG is non-differentiable because of
the hard quantisation step; we replace it with a smooth approximation so
gradients can flow from the decoder back through the attack layer into the
encoder.

Based on:
  - Shin & Song, "JPEG-resistant Adversarial Images", NeurIPS 2017 Workshop
  - Zhu et al., "HiDDeN: Hiding Data With Deep Networks", ECCV 2018

Pipeline:
  1. RGB → YCbCr colour transform  (linear, differentiable)
  2. 8×8 block DCT via learned / fixed circular convolutions
  3. Soft quantisation  (divide by Q-table, round via tanh approximation)
  4. De-quantisation    (multiply by Q-table)
  5. IDCT via transposed convolution
  6. YCbCr → RGB

Quality factor (QF) is adjustable at runtime (5–100).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ────────────────────────────────────────────────────────────
# Standard JPEG luminance & chrominance quantisation tables
# ────────────────────────────────────────────────────────────

_LUMINANCE_Q = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float32,
)

_CHROMINANCE_Q = torch.tensor(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=torch.float32,
)


def _quality_to_scale(qf: float) -> float:
    """Convert JPEG quality factor (1–100) to the IJG scale factor."""
    qf = max(1.0, min(100.0, qf))
    if qf < 50:
        return 5000.0 / qf
    return 200.0 - 2.0 * qf


def _scaled_qtable(base: torch.Tensor, qf: float) -> torch.Tensor:
    """Return the quantisation table scaled for the given quality factor."""
    s = _quality_to_scale(qf)
    table = torch.floor((base * s + 50.0) / 100.0)
    return table.clamp(min=1.0)


# ────────────────────────────────────────────────────────────
# DCT / IDCT basis via 8×8 convolution kernels
# ────────────────────────────────────────────────────────────


def _dct_basis() -> torch.Tensor:
    """Build the 64 DCT-II basis vectors as an (64, 1, 8, 8) conv kernel."""
    kernel = torch.zeros(64, 1, 8, 8)
    for u in range(8):
        for v in range(8):
            for x in range(8):
                for y in range(8):
                    val = math.cos((2 * x + 1) * u * math.pi / 16) * math.cos(
                        (2 * y + 1) * v * math.pi / 16
                    )
                    cu = 1.0 / math.sqrt(2) if u == 0 else 1.0
                    cv = 1.0 / math.sqrt(2) if v == 0 else 1.0
                    kernel[u * 8 + v, 0, x, y] = 0.25 * cu * cv * val
    return kernel


# ────────────────────────────────────────────────────────────
# Colour-space conversions  (RGB ↔ YCbCr, ITU-R BT.601)
# ────────────────────────────────────────────────────────────

# Matrices operate on (B, 3, H, W) via 1×1 convolution.
_RGB_TO_YCBCR = torch.tensor(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ],
    dtype=torch.float32,
)

_YCBCR_TO_RGB = torch.tensor(
    [
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0],
    ],
    dtype=torch.float32,
)

_YCBCR_OFFSET = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float32)


def _rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) RGB [0,1] → YCbCr [0,1]."""
    weight = _RGB_TO_YCBCR.to(x.device, x.dtype).unsqueeze(-1).unsqueeze(-1)  # (3,3,1,1)
    offset = _YCBCR_OFFSET.to(x.device, x.dtype).view(1, 3, 1, 1)
    return F.conv2d(x, weight) + offset


def _ycbcr_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) YCbCr [0,1] → RGB [0,1]."""
    weight = _YCBCR_TO_RGB.to(x.device, x.dtype).unsqueeze(-1).unsqueeze(-1)  # (3,3,1,1)
    offset = _YCBCR_OFFSET.to(x.device, x.dtype).view(1, 3, 1, 1)
    return F.conv2d(x - offset, weight)


# ────────────────────────────────────────────────────────────
# Differentiable rounding (STE + smooth approximation)
# ────────────────────────────────────────────────────────────


class _DiffRound(torch.autograd.Function):
    """Straight-Through Estimator for rounding.

    Forward: ``round(x)``
    Backward: identity (gradient of 1)
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        return x.round()

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor) -> torch.Tensor:
        return grad


def _soft_round(x: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Smooth rounding approximation using a periodic tanh.

    At low temperature this converges to ``round(x)``.  During training we
    use a moderate temperature (0.1–0.5) so gradients remain informative.
    """
    return x + 0.5 * torch.tanh(50.0 * temperature * (x - x.round()))


# ────────────────────────────────────────────────────────────
# Main module
# ────────────────────────────────────────────────────────────


class DifferentiableJPEG(nn.Module):
    """Fully-differentiable JPEG compression/decompression proxy.

    Parameters
    ----------
    quality : float
        Default quality factor (1–100).  Can be overridden per-call.
    use_ste : bool
        If *True* use the straight-through estimator for rounding (sharper
        but gradient = 1).  If *False* use the smooth ``tanh`` approximation
        (more accurate gradients but softer).
    temperature : float
        Temperature for the smooth rounding when ``use_ste=False``.
    """

    def __init__(
        self,
        quality: float = 50.0,
        use_ste: bool = False,
        temperature: float = 0.3,
    ) -> None:
        super().__init__()
        self.default_quality = quality
        self.use_ste = use_ste
        self.temperature = temperature

        # DCT / IDCT kernels — fixed, not trainable
        dct_kernel = _dct_basis()  # (64, 1, 8, 8)
        self.register_buffer("dct_kernel", dct_kernel)
        self.register_buffer("idct_kernel", dct_kernel)  # DCT-II is orthogonal → IDCT = transposed

        # Base quantisation tables
        self.register_buffer("lum_q", _LUMINANCE_Q)
        self.register_buffer("chr_q", _CHROMINANCE_Q)

    # ------------------------------------------------------------------
    def _block_dct(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 8×8 block DCT to a single-channel image (B, 1, H, W)."""
        return F.conv2d(x, self.dct_kernel, stride=8)  # (B, 64, H//8, W//8)

    def _block_idct(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Apply 8×8 block IDCT — transpose-convolution of the DCT kernel."""
        return F.conv_transpose2d(coeffs, self.idct_kernel, stride=8)  # (B, 1, H, W)

    # ------------------------------------------------------------------
    def _quantise(self, coeffs: torch.Tensor, qtable: torch.Tensor) -> torch.Tensor:
        """Divide by Q-table and round (differentiably)."""
        # qtable is (8, 8) → reshape to (1, 64, 1, 1) to broadcast
        q = qtable.reshape(1, 64, 1, 1)
        divided = coeffs / q
        if self.use_ste:
            return _DiffRound.apply(divided)
        return _soft_round(divided, self.temperature)

    def _dequantise(self, coeffs: torch.Tensor, qtable: torch.Tensor) -> torch.Tensor:
        """Multiply by Q-table (inverse of quantisation)."""
        q = qtable.reshape(1, 64, 1, 1)
        return coeffs * q

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        quality: float | None = None,
    ) -> torch.Tensor:
        """Compress and decompress ``x`` with differentiable JPEG.

        Parameters
        ----------
        x : Tensor
            Input RGB image tensor (B, 3, H, W) in [0, 1].
            H and W should be divisible by 8.
        quality : float | None
            Quality factor override (1–100).

        Returns
        -------
        Tensor
            Reconstructed image (B, 3, H, W) in [0, 1].
        """
        qf = quality if quality is not None else self.default_quality

        B, C, H, W = x.shape
        assert C == 3, f"Expected 3-channel RGB, got {C}"

        # Pad to multiples of 8 if needed
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, Hp, Wp = x.shape

        # 1. RGB → YCbCr  (shift range to [−0.5, 0.5] for DCT)
        ycbcr = _rgb_to_ycbcr(x) - 0.5  # centre around zero

        # 2. Scaled Q-tables for this quality factor
        lum_table = _scaled_qtable(self.lum_q, qf)
        chr_table = _scaled_qtable(self.chr_q, qf)

        channels = []
        for ch in range(3):
            c = ycbcr[:, ch : ch + 1, :, :]  # (B, 1, Hp, Wp)
            qt = lum_table if ch == 0 else chr_table

            # 3. Block DCT → quantise → dequantise → IDCT
            dct_coeffs = self._block_dct(c)
            quantised = self._quantise(dct_coeffs, qt)
            dequantised = self._dequantise(quantised, qt)
            reconstructed = self._block_idct(dequantised)
            channels.append(reconstructed)

        ycbcr_rec = torch.cat(channels, dim=1) + 0.5  # undo centre shift

        # 4. YCbCr → RGB
        rgb_rec = _ycbcr_to_rgb(ycbcr_rec)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            rgb_rec = rgb_rec[:, :, :H, :W]

        return rgb_rec.clamp(0.0, 1.0)
