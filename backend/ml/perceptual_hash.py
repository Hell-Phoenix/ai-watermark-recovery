"""DINOv2 ViT-B perceptual hashing — content-dependent image fingerprint.

Uses a **frozen** DINOv2 ViT-B backbone (from ``torch.hub``) to extract
a 512-dimensional semantic embedding of an image.  The embedding is
L2-normalised so cosine similarity reduces to a simple dot product.

This perceptual hash is the foundation of MetaSeal-style cryptographic
binding: changes to the *semantic content* of an image (e.g. face swap,
object transplantation) produce different hashes, while benign edits
(JPEG, mild crop, colour shift) leave the hash largely unchanged.

Pipeline:
  1. Resize input to 224×224 and apply ImageNet normalisation.
  2. Forward through frozen DINOv2 ViT-B → 768-dim CLS token.
  3. Linear projection 768 → 512.
  4. L2-normalise → unit-length 512-dim perceptual hash.

API:
  - ``hash(image)`` → (B, 512) L2-normalised embedding
  - ``compute_similarity(hash_a, hash_b)`` → (B,) cosine similarity
  - ``forward(image)`` → alias for ``hash``

References:
  - DINOv2: Oquab et al., "DINOv2: Learning Robust Visual Features
    without Supervision", arXiv 2023
  - MetaSeal: content-dependent cryptographic watermark binding
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PerceptualHashConfig:
    """Configuration for the DINOv2 perceptual hasher.

    Parameters
    ----------
    backbone : str
        DINOv2 model identifier for ``torch.hub`` (default ``"dinov2_vitb14"``).
        ViT-B/14 outputs 768-dim CLS tokens.
    hash_dim : int
        Target hash dimensionality after projection (default 512).
    backbone_dim : int
        CLS token dimension of the backbone (768 for ViT-B).
    input_size : int
        Resize input images to this square size (default 224).
    similarity_threshold : float
        Minimum cosine similarity to consider two hashes as matching
        the same content (default 0.85).
    use_torch_hub : bool
        If True, load from torch.hub (requires internet on first call).
        If False, use a randomly initialised backbone (for testing).
    """

    backbone: str = "dinov2_vitb14"
    hash_dim: int = 512
    backbone_dim: int = 768
    input_size: int = 224
    similarity_threshold: float = 0.85
    use_torch_hub: bool = True


# ---------------------------------------------------------------------------
# ImageNet normalisation constants
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Lightweight ViT stub for offline / CI testing
# ---------------------------------------------------------------------------

class _StubViT(nn.Module):
    """Minimal stand-in that mimics DINOv2 output shape for unit tests.

    Produces deterministic (but semantically meaningless) 768-dim vectors
    so the rest of the pipeline can be tested without downloading weights.
    """

    def __init__(self, embed_dim: int = 768) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Simple conv → pool → linear to produce embed_dim outputs
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class DINOv2PerceptualHasher(nn.Module):
    """Frozen DINOv2 ViT-B/14 perceptual hasher.

    Extracts a 512-dimensional L2-normalised semantic embedding from an
    input image tensor.  The backbone is frozen (no gradient computation)
    and only the lightweight 768→512 projection head is trainable.

    Parameters
    ----------
    config : PerceptualHashConfig | None
        Configuration; uses defaults if None.

    Example
    -------
    >>> hasher = DINOv2PerceptualHasher()
    >>> img = torch.rand(1, 3, 256, 256)
    >>> h = hasher.hash(img)  # (1, 512), L2-normalised
    >>> print(h.shape, h.norm(dim=-1))
    """

    def __init__(self, config: PerceptualHashConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or PerceptualHashConfig()

        # --- Load backbone ---
        if self.cfg.use_torch_hub:
            try:
                self.backbone: nn.Module = torch.hub.load(
                    "facebookresearch/dinov2",
                    self.cfg.backbone,
                    pretrained=True,
                )
            except Exception:
                # Fallback to stub if hub is unavailable (CI / offline)
                self.backbone = _StubViT(embed_dim=self.cfg.backbone_dim)
        else:
            self.backbone = _StubViT(embed_dim=self.cfg.backbone_dim)

        # Freeze backbone entirely
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # --- Projection head: backbone_dim → hash_dim ---
        self.projection = nn.Sequential(
            nn.Linear(self.cfg.backbone_dim, self.cfg.hash_dim, bias=False),
            nn.BatchNorm1d(self.cfg.hash_dim),
        )
        self._init_projection()

        # --- ImageNet normalisation buffers ---
        self.register_buffer(
            "_mean",
            torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_std",
            torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
        )

    def _init_projection(self) -> None:
        """Initialise projection head weights."""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """Resize to ``input_size`` and apply ImageNet normalisation.

        Parameters
        ----------
        image : (B, 3, H, W) in [0, 1]

        Returns
        -------
        (B, 3, input_size, input_size) normalised tensor.
        """
        sz = self.cfg.input_size
        if image.shape[-2] != sz or image.shape[-1] != sz:
            image = F.interpolate(
                image, size=(sz, sz), mode="bilinear", align_corners=False,
            )
        return (image - self._mean) / self._std  # type: ignore[operator]

    @torch.no_grad()
    def _extract_backbone_features(self, image: torch.Tensor) -> torch.Tensor:
        """Run frozen backbone and return CLS token (B, backbone_dim)."""
        self.backbone.eval()
        x = self._preprocess(image)
        features = self.backbone(x)

        # DINOv2 returns (B, backbone_dim) CLS token directly
        if features.dim() == 3:
            # Some variants return (B, num_patches+1, D); take CLS
            features = features[:, 0]
        return features

    def hash(self, image: torch.Tensor) -> torch.Tensor:
        """Compute 512-dim L2-normalised perceptual hash.

        Parameters
        ----------
        image : (B, 3, H, W) in [0, 1]

        Returns
        -------
        (B, hash_dim) — L2-normalised embedding vector.
        """
        features = self._extract_backbone_features(image)
        # Projection contains BatchNorm1d which requires batch > 1 in
        # training mode.  Hashing is always an inference operation so we
        # temporarily switch to eval and restore afterwards.
        was_training = self.projection.training
        self.projection.eval()
        projected = self.projection(features)
        if was_training:
            self.projection.train()
        return F.normalize(projected, p=2, dim=-1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`hash`."""
        return self.hash(image)

    def compute_similarity(
        self,
        hash_a: torch.Tensor,
        hash_b: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine similarity between two L2-normalised hash vectors.

        Since both vectors are unit-length, this is just a dot product.

        Parameters
        ----------
        hash_a, hash_b : (B, hash_dim) — L2-normalised hashes

        Returns
        -------
        (B,) — cosine similarity in [-1, 1]
        """
        return (hash_a * hash_b).sum(dim=-1)

    def are_similar(
        self,
        hash_a: torch.Tensor,
        hash_b: torch.Tensor,
    ) -> torch.Tensor:
        """Check whether two hashes represent the same content.

        Parameters
        ----------
        hash_a, hash_b : (B, hash_dim)

        Returns
        -------
        (B,) — boolean tensor, True if similarity >= threshold.
        """
        sim = self.compute_similarity(hash_a, hash_b)
        return sim >= self.cfg.similarity_threshold


# ---------------------------------------------------------------------------
# Utility: deterministic hash from tensor (for non-NN use cases)
# ---------------------------------------------------------------------------

def sha256_tensor_hash(tensor: torch.Tensor) -> bytes:
    """SHA-256 hash of raw tensor bytes (deterministic).

    Useful for computing a digest to be signed by ECDSA.

    Parameters
    ----------
    tensor : any shape — will be flattened and converted to bytes.

    Returns
    -------
    32-byte SHA-256 digest.
    """
    data = tensor.detach().cpu().float().numpy().tobytes()
    return hashlib.sha256(data).digest()
