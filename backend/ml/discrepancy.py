"""Dual-domain discrepancy detector and attack classifier.

Compares latent-layer and pixel-layer watermark extraction results to
classify the attack type applied to a watermarked image.  The core
insight: different attack families leave distinct *fingerprints* in the
agreement/disagreement between the two watermark layers.

Attack classification rules
----------------------------
- **CLEAN**     — both layers present, high bit agreement
- **JPEG**      — both layers degraded roughly equally
- **CROP**      — both layers degraded, latent slightly more robust
- **D2RA**      — latent layer intact, pixel layer scrubbed (diffusion-based
  regeneration attacks strip pixel-level watermarks while preserving
  semantic-level latent watermarks)
- **DAWN**      — similar to D2RA but latent partially degraded too
  (adversarial noise targets both layers, but pixel layer falls first)
- **DIFFUSION** — diffusion-based transformation or noise schedule
  modification that partially disrupts both layers

References:
  - D2RA attack: Zhao et al., 2025
  - DAWN attack: adversarial diffusion-based watermark removal
  - WIND: Wen et al., 2024 (latent-layer watermarking)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Attack taxonomy
# ---------------------------------------------------------------------------

class AttackType(str, Enum):
    """Enumeration of attack classes the discrepancy detector can identify."""

    CLEAN = "clean"
    JPEG = "jpeg"
    CROP = "crop"
    D2RA = "d2ra"            # Diffusion-based Regeneration Attack
    DAWN = "dawn"            # Adversarial watermark removal
    DIFFUSION = "diffusion"  # Diffusion-based noise / schedule attacks


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DiscrepancyConfig:
    """Hyper-parameters for the discrepancy detector.

    Attributes
    ----------
    message_bits : int
        Payload length shared by both watermark layers (default 48).
    latent_threshold : float
        Confidence threshold for latent layer to be considered present.
    pixel_threshold : float
        Confidence threshold for pixel layer to be considered present.
    agreement_threshold : float
        Bit-agreement ratio above which two extractions are considered
        consistent (used for CLEAN detection).
    d2ra_latent_min : float
        Minimum latent confidence to classify as D2RA (latent intact).
    d2ra_pixel_max : float
        Maximum pixel confidence to classify as D2RA (pixel scrubbed).
    dawn_latent_ratio : float
        If latent conf / pixel conf exceeds this, lean toward DAWN rather
        than generic degradation.
    classifier_hidden : int
        Hidden dimension for the learned classifier MLP.
    """

    message_bits: int = 48
    latent_threshold: float = 0.60
    pixel_threshold: float = 0.60
    agreement_threshold: float = 0.80
    d2ra_latent_min: float = 0.65
    d2ra_pixel_max: float = 0.45
    dawn_latent_ratio: float = 1.8
    classifier_hidden: int = 128


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _bit_agreement(bits_a: torch.Tensor, bits_b: torch.Tensor) -> torch.Tensor:
    """Per-sample fraction of agreeing bits.  (B, N) × (B, N) → (B,)."""
    return (bits_a == bits_b).float().mean(dim=-1)


def _bit_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample mean binary entropy of logit predictions.  (B, N) → (B,).

    Low entropy ≈ confident extraction, high entropy ≈ noise / scrubbed.
    """
    probs = torch.sigmoid(logits)
    eps = 1e-7
    entropy = -(probs * (probs + eps).log2() + (1 - probs) * (1 - probs + eps).log2())
    return entropy.mean(dim=-1)


def _confidence_gap(conf_a: torch.Tensor, conf_b: torch.Tensor) -> torch.Tensor:
    """Signed gap: positive when layer A is more confident."""
    return conf_a - conf_b


def build_feature_vector(
    latent_result: dict[str, torch.Tensor],
    pixel_result: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Stack handcrafted features into a (B, F) tensor for the classifier.

    Features (F = 7):
        0  latent_confidence
        1  pixel_confidence
        2  confidence_gap  (latent − pixel)
        3  bit_agreement   (between predicted bits of both layers)
        4  latent_entropy
        5  pixel_entropy
        6  entropy_gap     (pixel − latent, higher if pixel is more uncertain)
    """
    lat_conf = latent_result["confidence"]       # (B,)
    pix_conf = pixel_result["confidence"]        # (B,)
    lat_bits = latent_result["predicted_bits"]    # (B, N)
    pix_bits = pixel_result["predicted_bits"]     # (B, N)
    lat_logits = latent_result["logits"]          # (B, N)
    pix_logits = pixel_result["logits"]           # (B, N)

    agreement = _bit_agreement(lat_bits, pix_bits)
    lat_ent = _bit_entropy(lat_logits)
    pix_ent = _bit_entropy(pix_logits)

    features = torch.stack([
        lat_conf,
        pix_conf,
        _confidence_gap(lat_conf, pix_conf),
        agreement,
        lat_ent,
        pix_ent,
        pix_ent - lat_ent,
    ], dim=-1)  # (B, 7)

    return features


# ---------------------------------------------------------------------------
# Rule-based heuristic classifier
# ---------------------------------------------------------------------------

class RuleBasedDiscrepancy:
    """Stateless, threshold-based attack classifier (no learnable params).

    Works well for clear-cut scenarios; use :class:`LearnedDiscrepancy`
    when labelled attack data is available.
    """

    def __init__(self, config: DiscrepancyConfig | None = None) -> None:
        self.cfg = config or DiscrepancyConfig()

    def classify(
        self,
        latent_result: dict[str, torch.Tensor],
        pixel_result: dict[str, torch.Tensor],
    ) -> dict[str, object]:
        """Classify the attack type per sample.

        Returns
        -------
        dict with keys:
            attack_type : list[AttackType]
            confidence  : Tensor (B,) — overall detection confidence [0, 1]
            details     : dict      — intermediate diagnostics
        """
        lat_conf = latent_result["confidence"]
        pix_conf = pixel_result["confidence"]
        agreement = _bit_agreement(
            latent_result["predicted_bits"], pixel_result["predicted_bits"]
        )

        cfg = self.cfg
        B = lat_conf.shape[0]
        attacks: list[AttackType] = []

        for i in range(B):
            lc = lat_conf[i].item()
            pc = pix_conf[i].item()
            ag = agreement[i].item()

            if lc >= cfg.latent_threshold and pc >= cfg.pixel_threshold:
                if ag >= cfg.agreement_threshold:
                    attacks.append(AttackType.CLEAN)
                else:
                    # Both present but disagree → likely JPEG / noise
                    attacks.append(AttackType.JPEG)

            elif lc >= cfg.d2ra_latent_min and pc < cfg.d2ra_pixel_max:
                # Latent intact, pixel scrubbed → regeneration attack
                attacks.append(AttackType.D2RA)

            elif lc >= cfg.pixel_threshold and pc < cfg.pixel_threshold:
                # Latent partially intact, pixel gone
                ratio = lc / max(pc, 1e-6)
                if ratio >= cfg.dawn_latent_ratio:
                    attacks.append(AttackType.DAWN)
                else:
                    attacks.append(AttackType.CROP)

            elif lc < cfg.latent_threshold and pc < cfg.pixel_threshold:
                # Both scrubbed
                attacks.append(AttackType.CROP)

            else:
                # Catch-all for ambiguous cases: diffusion-like partial
                attacks.append(AttackType.DIFFUSION)

        # Confidence: clamp to [0, 1]
        overall_conf = ((lat_conf + pix_conf) / 2.0).clamp(0.0, 1.0)

        return {
            "attack_type": attacks,
            "confidence": overall_conf,
            "details": {
                "latent_confidence": lat_conf,
                "pixel_confidence": pix_conf,
                "bit_agreement": agreement,
            },
        }


# ---------------------------------------------------------------------------
# Learned MLP classifier
# ---------------------------------------------------------------------------

class LearnedDiscrepancy(nn.Module):
    """MLP-based attack classifier trained on (feature_vector → attack_type).

    Takes the 7-D feature vector from :func:`build_feature_vector` and
    outputs class logits over :class:`AttackType` categories.
    """

    NUM_CLASSES: int = len(AttackType)  # 6

    def __init__(self, config: DiscrepancyConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or DiscrepancyConfig()
        in_dim = 7  # feature vector length
        h = self.cfg.classifier_hidden

        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(h, h),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(h, self.NUM_CLASSES),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        latent_result: dict[str, torch.Tensor],
        pixel_result: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Predict attack class logits from dual-domain extraction results.

        Parameters
        ----------
        latent_result, pixel_result
            Dicts returned by ``LatentWatermarkDetector.extract()`` and
            ``IntegerInvertibleWatermarkNetwork.extract()``.

        Returns
        -------
        Tensor (B, NUM_CLASSES) — raw logits; apply softmax for probabilities.
            Class indices follow ``AttackType`` enum order:
            0=CLEAN, 1=JPEG, 2=CROP, 3=D2RA, 4=DAWN, 5=DIFFUSION
        """
        features = build_feature_vector(latent_result, pixel_result)
        return self.net(features)

    def predict(
        self,
        latent_result: dict[str, torch.Tensor],
        pixel_result: dict[str, torch.Tensor],
    ) -> dict[str, object]:
        """Friendly wrapper returning attack type labels + probabilities.

        Returns
        -------
        dict with keys:
            attack_type : list[AttackType]   — predicted class per sample
            probabilities : Tensor (B, NUM_CLASSES)
            confidence : Tensor (B,) — max probability per sample [0, 1]
        """
        logits = self.forward(latent_result, pixel_result)
        probs = F.softmax(logits, dim=-1)
        class_idx = logits.argmax(dim=-1)  # (B,)

        attack_list = list(AttackType)
        attacks = [attack_list[idx.item()] for idx in class_idx]
        confidence = probs.max(dim=-1).values  # already in [0, 1]

        return {
            "attack_type": attacks,
            "probabilities": probs,
            "confidence": confidence,
        }


# ---------------------------------------------------------------------------
# Loss for training the learned classifier
# ---------------------------------------------------------------------------

class DiscrepancyLoss(nn.Module):
    """Cross-entropy loss for attack classification, with optional class weights.

    Parameters
    ----------
    class_weights : Tensor (NUM_CLASSES,) | None
        Per-class weights to handle imbalanced attack distributions.
    label_smoothing : float
        Label smoothing factor (default 0.1).
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(LearnedDiscrepancy.NUM_CLASSES),
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute classification loss.

        Parameters
        ----------
        logits : Tensor (B, NUM_CLASSES)
        targets : Tensor (B,) — integer class indices [0, NUM_CLASSES)

        Returns
        -------
        dict with:
            loss     : scalar cross-entropy loss
            accuracy : scalar classification accuracy
        """
        loss = F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        preds = logits.argmax(dim=-1)
        accuracy = (preds == targets).float().mean()
        return {"loss": loss, "accuracy": accuracy}


# ---------------------------------------------------------------------------
# Unified dual-domain detector (combines extraction + classification)
# ---------------------------------------------------------------------------

class DualDomainDetector(nn.Module):
    """End-to-end dual-domain watermark detector.

    Wraps latent + pixel extractors and a discrepancy classifier into a
    single module for convenient inference.

    Parameters
    ----------
    latent_detector : nn.Module
        Must have an ``.extract(noise_latent, secret_key)`` method
        returning the standard result dict.
    pixel_extractor : nn.Module
        Must have an ``.extract(watermarked_image)`` method returning
        the standard result dict.
    use_learned_classifier : bool
        If True, use :class:`LearnedDiscrepancy`; otherwise fall back
        to :class:`RuleBasedDiscrepancy`.
    config : DiscrepancyConfig | None
        Shared configuration for thresholds and network dimensions.
    """

    def __init__(
        self,
        latent_detector: nn.Module,
        pixel_extractor: nn.Module,
        use_learned_classifier: bool = False,
        config: DiscrepancyConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or DiscrepancyConfig()
        self.latent_detector = latent_detector
        self.pixel_extractor = pixel_extractor

        if use_learned_classifier:
            self.classifier: nn.Module | RuleBasedDiscrepancy = LearnedDiscrepancy(self.cfg)
        else:
            self.classifier = RuleBasedDiscrepancy(self.cfg)

    def forward(
        self,
        image: torch.Tensor,
        z_T: torch.Tensor | None = None,
        secret_key: str = "default",
    ) -> dict[str, object]:
        """Run the full dual-domain detection + attack classification.

        Parameters
        ----------
        image : Tensor (B, 3, H, W)
            The possibly-attacked watermarked image.
        z_T : Tensor (B, C, H_lat, W_lat) | None
            Recovered noise latent for latent-layer detection.  If None,
            only pixel-layer detection is performed.
        secret_key : str
            Secret key used during latent embedding.

        Returns
        -------
        dict with keys:
            latent_result : dict — raw output from latent detector
            pixel_result  : dict — raw output from pixel extractor
            attack_type   : list[AttackType]
            confidence    : Tensor (B,) — in [0, 1]
            payload       : dict with merged bit predictions + fusion metadata
        """
        # --- Pixel layer extraction ---
        pixel_result = self.pixel_extractor.extract(image)

        # --- Latent layer extraction ---
        if z_T is not None:
            latent_result = self.latent_detector.extract(z_T, secret_key)
        else:
            # No latent available — fill with low-confidence placeholders
            B = image.shape[0]
            N = self.cfg.message_bits
            device = image.device
            latent_result = {
                "logits": torch.zeros(B, N, device=device),
                "predicted_bits": torch.zeros(B, N, device=device),
                "confidence": torch.zeros(B, device=device),
                "is_watermarked": torch.zeros(B, dtype=torch.bool, device=device),
            }

        # --- Discrepancy classification ---
        if isinstance(self.classifier, LearnedDiscrepancy):
            classification = self.classifier.predict(latent_result, pixel_result)
        else:
            classification = self.classifier.classify(latent_result, pixel_result)

        # --- Payload fusion ---
        payload = self._fuse_payloads(latent_result, pixel_result)

        return {
            "latent_result": latent_result,
            "pixel_result": pixel_result,
            "attack_type": classification["attack_type"],
            "confidence": classification["confidence"],
            "payload": payload,
        }

    def _fuse_payloads(
        self,
        latent_result: dict[str, torch.Tensor],
        pixel_result: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Fuse bit predictions from both layers using confidence weighting.

        The more confident layer dominates the final payload.  In case of
        strong agreement, confidence is boosted.

        Returns
        -------
        dict with:
            fused_bits : Tensor (B, N)   — hard bit decisions
            fused_confidence : Tensor (B,) — fused confidence in [0, 1]
            source     : str             — "dual" | "latent_only" | "pixel_only"
        """
        lat_conf = latent_result["confidence"]
        pix_conf = pixel_result["confidence"]
        lat_logits = latent_result["logits"]
        pix_logits = pixel_result["logits"]

        # Confidence-weighted fusion of logits
        lat_w = lat_conf.unsqueeze(-1)  # (B, 1)
        pix_w = pix_conf.unsqueeze(-1)
        total_w = lat_w + pix_w + 1e-8
        fused_logits = (lat_w * lat_logits + pix_w * pix_logits) / total_w

        fused_bits = (fused_logits > 0).float()

        # Fused confidence: geometric mean boosted by agreement, clamped to [0, 1]
        agreement = _bit_agreement(
            latent_result["predicted_bits"], pixel_result["predicted_bits"]
        )
        base_conf = (lat_conf * pix_conf).sqrt()
        fused_conf = (base_conf * (0.5 + 0.5 * agreement)).clamp(0.0, 1.0)

        # Determine source label
        lat_present = (lat_conf > self.cfg.latent_threshold).all().item()
        pix_present = (pix_conf > self.cfg.pixel_threshold).all().item()
        if lat_present and pix_present:
            source = "dual"
        elif lat_present:
            source = "latent_only"
        elif pix_present:
            source = "pixel_only"
        else:
            source = "none"

        return {
            "fused_bits": fused_bits,
            "fused_confidence": fused_conf,
            "source": source,
        }
