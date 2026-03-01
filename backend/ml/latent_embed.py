"""WIND-style latent watermarking — Fourier pattern injection into noise latent.

Implements the core idea from WIND (Wei et al., ICLR 2025): embed a
watermark by injecting a deterministic Fourier-domain pattern into the
initial Gaussian noise z used for diffusion sampling.  The key insight
is that properly constructed Fourier patterns preserve the statistical
properties of Gaussian noise (pass Kolmogorov–Smirnov tests), making
the watermark **undetectable** by distribution tests while remaining
recoverable via DDIM inversion.

Pipeline:
  1. **Key derivation** — HMAC-SHA256(secret_key ‖ "basis") → 256-bit seed
  2. **Pattern generation** — seed → PRNG → Fourier coefficients →
     inverse FFT → spatial-domain carrier patterns (one per bit)
  3. **Embedding** — z_w = renorm(z + α · Σ bᵢ · Pᵢ)
  4. **Extraction** — given recovered z̃_T, compute per-bit correlation
     with each carrier pattern to decode the embedded user_id bits

The pattern P is designed so that z_w remains i.i.d. N(0, 1) in
distribution.  This is achieved by:
  - Generating P in the Fourier domain with controlled magnitude
  - Per-channel renormalisation after injection
  - Using orthogonal frequency bins to minimise cross-correlation

API:
  - ``embed(noise_latent, secret_key, user_id)`` → (z_w, info)
  - ``extract(noise_latent, secret_key)`` → dict

References:
  - WIND: Wei et al., "Dreaming Watermarks: Hard Distortion-Free
    Watermarks in a Generated Image", ICLR 2025
  - Tree-Ring: Wen et al., "Tree-Ring Watermarks", NeurIPS 2023
  - Gaussian Shading: Yang et al., 2024
"""

from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

def hmac_sha256(salt: bytes, identifier: bytes) -> bytes:
    """Compute HMAC-SHA256(salt ‖ identifier) → 32-byte key."""
    return hmac.new(salt, identifier, hashlib.sha256).digest()


def derive_seed(salt: str | bytes, identifier: str | bytes) -> int:
    """Derive a deterministic integer seed from (salt, identifier).

    Uses HMAC-SHA256 to produce a 256-bit digest, then converts the
    first 8 bytes to a 64-bit integer suitable for ``torch.Generator``.
    """
    if isinstance(salt, str):
        salt = salt.encode("utf-8")
    if isinstance(identifier, str):
        identifier = identifier.encode("utf-8")
    digest = hmac_sha256(salt, identifier)
    return int.from_bytes(digest[:8], byteorder="big")


def _user_id_to_bits(user_id: str, num_bits: int) -> torch.Tensor:
    """Convert a user_id string to a binary tensor of fixed length.

    Uses SHA-256 hash of the user_id to produce deterministic bits.

    Returns
    -------
    Tensor (num_bits,) — binary {0, 1} float tensor.
    """
    digest = hashlib.sha256(user_id.encode("utf-8")).digest()
    # Convert bytes → bit array (big-endian)
    all_bits: list[int] = []
    for byte_val in digest:
        for bit_pos in range(7, -1, -1):
            all_bits.append((byte_val >> bit_pos) & 1)
    # SHA-256 yields 256 bits; tile if needed, truncate to num_bits
    while len(all_bits) < num_bits:
        all_bits = all_bits + all_bits
    return torch.tensor(all_bits[:num_bits], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Fourier pattern generator
# ---------------------------------------------------------------------------

class FourierPatternGenerator(nn.Module):
    """Generate a Fourier-domain watermark pattern from a key seed.

    The pattern is deterministic given the seed and has the property
    that adding it (scaled) to Gaussian noise preserves the Gaussian
    distribution.

    Parameters
    ----------
    latent_shape : tuple[int, ...]
        Shape of the noise latent, e.g. (C, H, W) = (3, 64, 64) for
        pixel-space or (4, 64, 64) for SD latent space.
    message_bits : int
        Number of watermark bits to encode (default 48).
    pattern_scale : float
        Base magnitude of the Fourier coefficients (default 1.0).
        The actual embedding strength is controlled by ``alpha`` at
        injection time.
    """

    def __init__(
        self,
        latent_shape: tuple[int, ...] = (3, 64, 64),
        message_bits: int = 48,
        pattern_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_shape = latent_shape
        self.message_bits = message_bits
        self.pattern_scale = pattern_scale

    def _seed_to_generator(self, seed: int, device: torch.device) -> torch.Generator:
        """Create a seeded ``torch.Generator`` on the given device."""
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        return gen

    def generate_carrier(
        self,
        seed: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Generate a unit-variance Fourier carrier pattern from *seed*.

        The carrier is generated in frequency space:
          1. Sample random phase angles from the seeded PRNG
          2. Assign uniform magnitude (controlled later by alpha)
          3. Inverse FFT → spatial pattern
          4. Normalise to unit variance

        Returns
        -------
        Tensor (*latent_shape) — spatial-domain carrier pattern with
        zero mean and unit variance.
        """
        gen = self._seed_to_generator(seed, device)
        C, H, W = self.latent_shape

        # Generate random phases in Fourier space
        phases = torch.rand(C, H, W, generator=gen, device=device, dtype=dtype) * 2 * math.pi

        # Uniform magnitude in frequency domain
        magnitude = torch.ones(C, H, W, device=device, dtype=dtype) * self.pattern_scale

        # Construct complex Fourier coefficients
        fourier_real = magnitude * torch.cos(phases)
        fourier_imag = magnitude * torch.sin(phases)
        fourier = torch.complex(fourier_real, fourier_imag)

        # Inverse FFT to spatial domain
        pattern = torch.fft.ifft2(fourier).real  # (C, H, W)

        # Normalise to zero-mean, unit-variance
        pattern = pattern - pattern.mean()
        std = pattern.std()
        if std > 1e-8:
            pattern = pattern / std

        return pattern

    def encode_message(
        self,
        message: torch.Tensor,
        seed: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Encode a binary message into a Fourier pattern.

        Each bit is assigned a group of Fourier frequency bins.  A bit
        value of 1 adds positive magnitude to that group; 0 adds
        negative magnitude.

        Parameters
        ----------
        message : Tensor (N,) — binary message (0/1 floats)
        seed : int — PRNG seed for frequency-bin assignment
        device : torch device
        dtype : torch dtype

        Returns
        -------
        Tensor (*latent_shape) — message-carrying pattern, unit variance.
        """
        gen = self._seed_to_generator(seed, device)
        C, H, W = self.latent_shape
        N = self.message_bits
        assert message.shape[-1] == N, f"Expected {N} bits, got {message.shape[-1]}"

        # Create a random assignment of frequency bins to message bits
        total_bins = C * H * W
        assignment = torch.randint(0, N, (total_bins,), generator=gen, device=device)

        # Build Fourier magnitude: +1 for bit=1, -1 for bit=0
        # Map message {0, 1} → {-1, +1}
        bit_signs = 2.0 * message - 1.0  # (N,)
        bin_signs = bit_signs[assignment].view(C, H, W)  # (C, H, W)

        # Random phases (same seed → reproducible)
        gen2 = self._seed_to_generator(seed + 1, device)
        phases = torch.rand(C, H, W, generator=gen2, device=device, dtype=dtype) * 2 * math.pi

        # Fourier pattern with message-modulated magnitude
        fourier_real = bin_signs * self.pattern_scale * torch.cos(phases)
        fourier_imag = bin_signs * self.pattern_scale * torch.sin(phases)
        fourier = torch.complex(fourier_real.to(dtype), fourier_imag.to(dtype))

        pattern = torch.fft.ifft2(fourier).real  # (C, H, W)

        # Normalise
        pattern = pattern - pattern.mean()
        std = pattern.std()
        if std > 1e-8:
            pattern = pattern / std

        return pattern

    def decode_message(
        self,
        z_T: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        """Decode watermark bits from a recovered noise latent.

        Performs the inverse of ``encode_message``: computes per-bit
        correlation by grouping frequency bins and averaging their
        signs.

        Parameters
        ----------
        z_T : Tensor (*latent_shape) or (B, *latent_shape)
        seed : int — same seed used for encoding

        Returns
        -------
        Tensor (B, N) or (N,) — decoded bit logits (positive → 1, negative → 0)
        """
        single = z_T.dim() == len(self.latent_shape)
        if single:
            z_T = z_T.unsqueeze(0)

        B = z_T.shape[0]
        C, H, W = self.latent_shape
        N = self.message_bits
        device = z_T.device

        # Reconstruct the frequency-bin assignment
        gen = self._seed_to_generator(seed, device)
        total_bins = C * H * W
        assignment = torch.randint(0, N, (total_bins,), generator=gen, device=device)

        # Reconstruct random phases
        gen2 = self._seed_to_generator(seed + 1, device)
        phases = torch.rand(C, H, W, generator=gen2, device=device, dtype=z_T.dtype) * 2 * math.pi

        # Compute correlation per frequency bin with the expected phase
        # Reference direction (unit vector for each bin)
        ref_real = torch.cos(phases).view(1, -1)  # (1, total_bins)
        ref_imag = torch.sin(phases).view(1, -1)

        # FFT of the recovered latent
        z_fft = torch.fft.fft2(z_T)  # (B, C, H, W) complex
        z_real = z_fft.real.view(B, -1)  # (B, total_bins)
        z_imag = z_fft.imag.view(B, -1)

        # Signed projection onto reference direction
        projection = z_real * ref_real + z_imag * ref_imag  # (B, total_bins)

        # Average projection per bit group
        logits = torch.zeros(B, N, device=device, dtype=z_T.dtype)
        for bit_idx in range(N):
            mask = assignment == bit_idx
            if mask.any():
                logits[:, bit_idx] = projection[:, mask].mean(dim=1)

        if single:
            logits = logits.squeeze(0)
        return logits

    def forward(
        self,
        z: torch.Tensor,
        message: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        """Encode message into a Fourier pattern and add to z.

        Parameters
        ----------
        z : Tensor (*latent_shape) — input noise
        message : Tensor (N,) — binary message
        seed : int — PRNG seed

        Returns
        -------
        Tensor — message-carrying pattern added to z.
        """
        pattern = self.encode_message(message, seed, device=z.device, dtype=z.dtype)
        return z + pattern


# ---------------------------------------------------------------------------
# Main embedding / detection module
# ---------------------------------------------------------------------------

@dataclass
class LatentWatermarkConfig:
    """Configuration for WIND-style latent watermarking."""

    # Noise latent shape (channels, height, width)
    latent_channels: int = 3
    latent_size: int = 64

    # Watermark
    message_bits: int = 48
    alpha: float = 2.5  # embedding strength
    pattern_scale: float = 1.0

    # Detection
    detection_threshold: float = 0.6  # min confidence for "detected"


class LatentWatermarkEmbedder(nn.Module):
    """WIND-style latent watermark embedder.

    Injects a Fourier-domain pattern into Gaussian noise z so that
    z_w = renorm(z + α · P(user_id, seed)).  The resulting z_w is
    statistically close to N(0, 1) while carrying a recoverable payload
    encoding the user_id.

    Usage
    -----
    >>> embedder = LatentWatermarkEmbedder()
    >>> z = torch.randn(1, 3, 64, 64)
    >>> z_w, info = embedder.embed(z, secret_key="my-key", user_id="user-42")
    """

    def __init__(self, config: LatentWatermarkConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or LatentWatermarkConfig()
        self.pattern_gen = FourierPatternGenerator(
            latent_shape=(self.cfg.latent_channels, self.cfg.latent_size, self.cfg.latent_size),
            message_bits=self.cfg.message_bits,
            pattern_scale=self.cfg.pattern_scale,
        )

    def embed(
        self,
        noise_latent: torch.Tensor,
        secret_key: str,
        user_id: str,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """Embed watermark into noise latent.

        Parameters
        ----------
        noise_latent : (B, C, H, W) or (C, H, W) — standard Gaussian noise
        secret_key : str — shared secret for Fourier basis derivation
        user_id : str — user identifier to be encoded as watermark payload

        Returns
        -------
        z_w : same shape as noise_latent — watermarked noise (still ≈ Gaussian)
        info : dict with ``pattern``, ``seed``, ``alpha``, ``snr``, ``user_bits``
        """
        single = noise_latent.dim() == 3
        if single:
            noise_latent = noise_latent.unsqueeze(0)

        B = noise_latent.shape[0]
        seed = derive_seed(secret_key, "basis")

        # Convert user_id → binary payload
        user_bits = _user_id_to_bits(user_id, self.cfg.message_bits).to(
            device=noise_latent.device
        )

        # Generate message-carrying Fourier pattern for each sample
        patterns = []
        for _ in range(B):
            p = self.pattern_gen.encode_message(
                user_bits, seed, device=noise_latent.device, dtype=noise_latent.dtype
            )
            patterns.append(p)
        pattern = torch.stack(patterns)  # (B, C, H, W)

        # Inject: z_w = z + α · P
        alpha = self.cfg.alpha
        z_w = noise_latent + alpha * pattern

        # Renormalise to preserve unit variance per channel
        # This ensures z_w passes statistical tests for Gaussianity
        for c in range(z_w.shape[1]):
            ch = z_w[:, c]
            ch_mean = ch.mean(dim=(-2, -1), keepdim=True)
            ch_std = ch.std(dim=(-2, -1), keepdim=True)
            z_w[:, c] = (ch - ch_mean) / (ch_std + 1e-8)

        # Compute SNR for diagnostics
        signal_power = (alpha * pattern).pow(2).mean()
        noise_power = noise_latent.pow(2).mean()
        snr = 10.0 * torch.log10(signal_power / (noise_power + 1e-10))

        if single:
            z_w = z_w.squeeze(0)
            pattern = pattern.squeeze(0)

        return z_w, {
            "pattern": pattern.detach(),
            "seed": seed,
            "alpha": alpha,
            "snr": snr.detach(),
            "user_bits": user_bits.detach(),
        }

    def forward(
        self,
        noise_latent: torch.Tensor,
        secret_key: str,
        user_id: str,
    ) -> torch.Tensor:
        """Embed and return watermarked noise only."""
        z_w, _ = self.embed(noise_latent, secret_key, user_id)
        return z_w


class LatentWatermarkDetector(nn.Module):
    """Detect and extract WIND-style latent watermark from recovered z_T.

    Pairs with ``LatentWatermarkEmbedder``: given a noise latent
    recovered via DDIM inversion, extracts the embedded user_id bits
    and computes confidence / detection metrics.
    """

    def __init__(self, config: LatentWatermarkConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or LatentWatermarkConfig()
        self.pattern_gen = FourierPatternGenerator(
            latent_shape=(self.cfg.latent_channels, self.cfg.latent_size, self.cfg.latent_size),
            message_bits=self.cfg.message_bits,
            pattern_scale=self.cfg.pattern_scale,
        )

    def extract(
        self,
        noise_latent: torch.Tensor,
        secret_key: str,
    ) -> dict[str, torch.Tensor]:
        """Extract watermark bits from a recovered noise latent.

        Parameters
        ----------
        noise_latent : (B, C, H, W) or (C, H, W) — recovered noise latent
        secret_key : str — same secret key used during embedding

        Returns
        -------
        dict with:
          - ``logits`` : (B, N) raw decoder logits
          - ``predicted_bits`` : (B, N) hard decisions
          - ``confidence`` : (B,) detection confidence in [0, 1]
          - ``is_watermarked`` : (B,) boolean detection decision
        """
        single = noise_latent.dim() == 3
        if single:
            noise_latent = noise_latent.unsqueeze(0)

        seed = derive_seed(secret_key, "basis")
        logits = self.pattern_gen.decode_message(noise_latent, seed)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        predicted_bits = (logits > 0).float()
        confidence = logits.abs().mean(dim=1)

        # Normalise confidence to [0, 1] range using sigmoid mapping
        confidence_01 = torch.sigmoid(confidence - 0.5)

        result: dict[str, torch.Tensor] = {
            "logits": logits,
            "predicted_bits": predicted_bits,
            "confidence": confidence_01,
            "is_watermarked": confidence_01 > self.cfg.detection_threshold,
        }

        if single:
            result = {k: v.squeeze(0) if v.dim() > 1 else v for k, v in result.items()}

        return result

    def forward(
        self,
        noise_latent: torch.Tensor,
        secret_key: str,
    ) -> torch.Tensor:
        """Extract predicted bits from recovered noise."""
        result = self.extract(noise_latent, secret_key)
        return result["predicted_bits"]


# ---------------------------------------------------------------------------
# Gaussianity verification (test utility)
# ---------------------------------------------------------------------------

def ks_test_gaussianity(z: torch.Tensor) -> dict[str, float]:
    """Run Kolmogorov–Smirnov test for Gaussianity on flattened tensor.

    Uses PyTorch-only implementation (no scipy dependency).

    Returns
    -------
    dict with ``ks_statistic``, ``mean``, ``std``, ``skewness``,
    ``kurtosis`` (excess).
    """
    flat = z.flatten().float()
    n = flat.numel()

    # Sort values
    sorted_vals = flat.sort()[0]

    # Standardise
    mu = sorted_vals.mean()
    sigma = sorted_vals.std()
    standardised = (sorted_vals - mu) / (sigma + 1e-10)

    # Empirical CDF
    ecdf = torch.arange(1, n + 1, device=z.device, dtype=torch.float32) / n

    # Standard normal CDF (approximation using erfc)
    from_std_normal = 0.5 * (1 + torch.erf(standardised / math.sqrt(2)))

    # KS statistic = max|ECDF - CDF|
    ks_stat = (ecdf - from_std_normal).abs().max().item()

    # Higher moments
    skewness = ((standardised ** 3).mean()).item()
    kurtosis = ((standardised ** 4).mean() - 3.0).item()  # excess

    return {
        "ks_statistic": ks_stat,
        "mean": mu.item(),
        "std": sigma.item(),
        "skewness": skewness,
        "kurtosis": kurtosis,
    }
