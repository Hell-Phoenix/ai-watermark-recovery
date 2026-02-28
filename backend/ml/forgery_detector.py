"""Forgery / transplantation attack detector.

Detects when a watermark has been **transplanted** from one image to
another by comparing the freshly computed perceptual hash of the test
image against the ECDSA signature extracted from the watermark payload.

Detection logic:
  1. Extract watermark payload → embedded ECDSA signature bits (512 bits)
  2. Recompute DINOv2 perceptual hash of the test image → 512-dim vector
  3. Verify: ``ECDSA_verify(SHA-256(hash), signature, public_key)``
  4. If verification **fails** → the watermark was created for a different
     image → **transplantation / forgery detected**

Additionally computes a continuous *content fidelity score* via cosine
similarity between the recomputed hash and a reference hash (if provided),
giving a soft measure of how much the image has changed.

API:
  - ``detect(image, signature, public_key, ...)`` → ForgeryResult
  - ``detect_from_bits(image, signature_bits, public_key, ...)`` → ForgeryResult

References:
  - MetaSeal: content-dependent cryptographic watermark binding
  - DINOv2 perceptual hashing: see ``perceptual_hash.py``
  - ECDSA verification: see ``ecdsa_signer.py``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from backend.ml.perceptual_hash import DINOv2PerceptualHasher, PerceptualHashConfig
from backend.ml.ecdsa_signer import (
    ECDSAConfig,
    ECDSAKeyPair,
    ECDSASigner,
    ECDSAVerifier,
    bits_to_signature,
    signature_to_bits,
    _embedding_to_digest,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ForgeryDetectorConfig:
    """Configuration for the forgery detector.

    Parameters
    ----------
    hash_config : PerceptualHashConfig
        Configuration for the DINOv2 hasher.
    ecdsa_config : ECDSAConfig
        Configuration for ECDSA verification.
    similarity_threshold : float
        Minimum cosine similarity between recomputed and reference hash
        to consider the content unchanged (default 0.85).
    strict_mode : bool
        If True, *any* ECDSA verification failure → forgery.
        If False, use soft scoring combining ECDSA + cosine similarity.
    """

    hash_config: PerceptualHashConfig = field(
        default_factory=lambda: PerceptualHashConfig(use_torch_hub=False),
    )
    ecdsa_config: ECDSAConfig = field(default_factory=ECDSAConfig)
    similarity_threshold: float = 0.85
    strict_mode: bool = True


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ForgeryResult:
    """Output of the forgery detection pipeline.

    Attributes
    ----------
    forgery_detected : bool
        True if transplantation / forgery is flagged.
    ecdsa_valid : bool
        True if ECDSA signature verification passed.
    content_similarity : float
        Cosine similarity between recomputed hash and reference
        (1.0 if no reference provided — assumes match).
    confidence : float
        Overall confidence in the forgery verdict, in [0, 1].
        High confidence + forgery_detected → strong forgery evidence.
    recomputed_hash : torch.Tensor
        The freshly computed perceptual hash (512-dim).
    details : dict
        Additional diagnostic information.
    """

    forgery_detected: bool
    ecdsa_valid: bool
    content_similarity: float
    confidence: float
    recomputed_hash: torch.Tensor
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Forgery detector
# ---------------------------------------------------------------------------

class ForgeryDetector(nn.Module):
    """End-to-end transplantation attack detector.

    Wraps a :class:`DINOv2PerceptualHasher` and :class:`ECDSAVerifier`
    to detect watermark transplantation from one image to another.

    Parameters
    ----------
    config : ForgeryDetectorConfig | None
        Configuration; uses defaults if None.
    hasher : DINOv2PerceptualHasher | None
        Pre-built hasher instance. If None, one is created from config.
    """

    def __init__(
        self,
        config: ForgeryDetectorConfig | None = None,
        hasher: DINOv2PerceptualHasher | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or ForgeryDetectorConfig()
        self.hasher = hasher or DINOv2PerceptualHasher(self.cfg.hash_config)

    def detect(
        self,
        image: torch.Tensor,
        signature: bytes,
        public_key: "ec.EllipticCurvePublicKey",  # type: ignore[name-defined]
        reference_hash: Optional[torch.Tensor] = None,
    ) -> ForgeryResult:
        """Detect whether the watermark in *image* was transplanted.

        Parameters
        ----------
        image : (B, 3, H, W) or (1, 3, H, W) in [0, 1]
            The test image to verify.
        signature : bytes
            64-byte compact ECDSA signature extracted from watermark.
        public_key : ec.EllipticCurvePublicKey
            Public key of the original signer.
        reference_hash : (512,) or (1, 512) | None
            If provided, compute cosine similarity against this
            reference hash for a soft content match score.

        Returns
        -------
        ForgeryResult with all diagnostic fields populated.
        """
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # 1. Recompute perceptual hash of the test image
        recomputed = self.hasher.hash(image)  # (B, 512)
        # Use first sample for ECDSA (single-image verification)
        recomputed_single = recomputed[0]  # (512,)

        # 2. ECDSA verification
        verifier = ECDSAVerifier(public_key)
        ecdsa_valid = verifier.verify_embedding(recomputed_single, signature)

        # 3. Content similarity (if reference provided)
        if reference_hash is not None:
            ref = reference_hash.detach()
            if ref.dim() > 1:
                ref = ref.squeeze(0)
            content_sim = float(
                (recomputed_single * ref).sum().item()
            )
        else:
            content_sim = 1.0  # no reference = assume match

        # 4. Determine forgery
        if self.cfg.strict_mode:
            forgery_detected = not ecdsa_valid
            confidence = 1.0 if not ecdsa_valid else 0.0
        else:
            # Soft scoring: combine ECDSA result and content similarity
            ecdsa_score = 0.0 if ecdsa_valid else 1.0
            content_score = max(0.0, 1.0 - content_sim)  # higher = more different
            # Weighted combination
            forgery_score = 0.7 * ecdsa_score + 0.3 * content_score
            forgery_detected = forgery_score > 0.5
            confidence = forgery_score

        return ForgeryResult(
            forgery_detected=forgery_detected,
            ecdsa_valid=ecdsa_valid,
            content_similarity=content_sim,
            confidence=confidence,
            recomputed_hash=recomputed_single.detach(),
            details={
                "strict_mode": self.cfg.strict_mode,
                "similarity_threshold": self.cfg.similarity_threshold,
                "batch_hashes": recomputed.detach(),
            },
        )

    def detect_from_bits(
        self,
        image: torch.Tensor,
        signature_bits: torch.Tensor,
        public_key: "ec.EllipticCurvePublicKey",  # type: ignore[name-defined]
        reference_hash: Optional[torch.Tensor] = None,
    ) -> ForgeryResult:
        """Same as :meth:`detect` but accepts signature as binary tensor.

        Parameters
        ----------
        image : (B, 3, H, W) in [0, 1]
        signature_bits : (512,) — binary {0, 1}
        public_key : ec.EllipticCurvePublicKey
        reference_hash : optional (512,) tensor

        Returns
        -------
        ForgeryResult
        """
        sig_bytes = bits_to_signature(signature_bits)
        return self.detect(image, sig_bytes, public_key, reference_hash)

    def forward(
        self,
        image: torch.Tensor,
        signature_bits: torch.Tensor,
        public_key: "ec.EllipticCurvePublicKey",  # type: ignore[name-defined]
    ) -> dict[str, object]:
        """Forward pass returning a dict (for integration with pipelines).

        Parameters
        ----------
        image : (B, 3, H, W)
        signature_bits : (512,) binary tensor
        public_key : ec.EllipticCurvePublicKey

        Returns
        -------
        dict with ``forgery_detected``, ``ecdsa_valid``, ``confidence``,
        ``content_similarity``, ``recomputed_hash``.
        """
        result = self.detect_from_bits(image, signature_bits, public_key)
        return {
            "forgery_detected": result.forgery_detected,
            "ecdsa_valid": result.ecdsa_valid,
            "confidence": result.confidence,
            "content_similarity": result.content_similarity,
            "recomputed_hash": result.recomputed_hash,
        }


# ---------------------------------------------------------------------------
# Full MetaSeal pipeline: sign + embed, then verify + detect
# ---------------------------------------------------------------------------

class MetaSealPipeline:
    """High-level pipeline for content-dependent cryptographic binding.

    Combines perceptual hashing, ECDSA signing, and forgery detection
    into a single coherent API.

    Parameters
    ----------
    hasher : DINOv2PerceptualHasher
        Perceptual hash extractor.
    key_pair : ECDSAKeyPair
        ECDSA key pair for signing (private) and verification (public).
    config : ForgeryDetectorConfig | None
        Detector configuration.
    """

    def __init__(
        self,
        hasher: DINOv2PerceptualHasher,
        key_pair: ECDSAKeyPair,
        config: ForgeryDetectorConfig | None = None,
    ) -> None:
        self.hasher = hasher
        self.key_pair = key_pair
        self.signer = ECDSASigner(key_pair)
        self.verifier = ECDSAVerifier(key_pair.public_key)
        self.detector = ForgeryDetector(
            config=config, hasher=hasher,
        )

    def sign_image(self, image: torch.Tensor) -> dict[str, object]:
        """Compute perceptual hash and ECDSA signature for an image.

        Parameters
        ----------
        image : (1, 3, H, W) or (3, H, W) in [0, 1]

        Returns
        -------
        dict with:
          - ``hash``: (512,) L2-normalised perceptual hash
          - ``signature``: 64-byte compact ECDSA signature
          - ``signature_bits``: (512,) binary tensor for watermark embedding
          - ``signature_hex``: hex-encoded signature string
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        perceptual_hash = self.hasher.hash(image)[0]  # (512,)
        signature = self.signer.sign_embedding(perceptual_hash)
        sig_bits = signature_to_bits(signature)

        return {
            "hash": perceptual_hash,
            "signature": signature,
            "signature_bits": sig_bits,
            "signature_hex": signature.hex(),
        }

    def verify_image(
        self,
        image: torch.Tensor,
        signature: bytes,
        reference_hash: Optional[torch.Tensor] = None,
    ) -> ForgeryResult:
        """Verify an image against an extracted signature.

        Parameters
        ----------
        image : (1, 3, H, W) or (3, H, W) in [0, 1]
        signature : 64-byte compact signature
        reference_hash : optional (512,) for similarity comparison

        Returns
        -------
        ForgeryResult
        """
        return self.detector.detect(
            image, signature, self.key_pair.public_key,
            reference_hash=reference_hash,
        )

    def verify_image_from_bits(
        self,
        image: torch.Tensor,
        signature_bits: torch.Tensor,
        reference_hash: Optional[torch.Tensor] = None,
    ) -> ForgeryResult:
        """Verify using a binary signature tensor."""
        return self.detector.detect_from_bits(
            image, signature_bits, self.key_pair.public_key,
            reference_hash=reference_hash,
        )
