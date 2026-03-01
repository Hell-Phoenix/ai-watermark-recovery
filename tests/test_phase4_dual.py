"""Phase 4 — dual-domain watermarking tests.

Covers latent_embed, pixel_embed, iiwn, adn, crypto, and discrepancy modules.
"""

from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# Latent embedding (WIND-style)
# ---------------------------------------------------------------------------
from backend.ml.latent_embed import (
    FourierPatternGenerator,
    LatentWatermarkConfig,
    LatentWatermarkDetector,
    LatentWatermarkEmbedder,
    _user_id_to_bits,
    ks_test_gaussianity,
)


class TestFourierPatternGenerator:
    def test_pattern_shape(self) -> None:
        gen = FourierPatternGenerator(latent_shape=(3, 64, 64))
        pattern = gen.generate_carrier(seed=42)
        assert pattern.shape == (3, 64, 64)

    def test_deterministic(self) -> None:
        gen = FourierPatternGenerator(latent_shape=(3, 64, 64))
        p1 = gen.generate_carrier(seed=123)
        p2 = gen.generate_carrier(seed=123)
        assert torch.allclose(p1, p2)

    def test_different_ids_differ(self) -> None:
        gen = FourierPatternGenerator(latent_shape=(3, 64, 64))
        p1 = gen.generate_carrier(seed=1)
        p2 = gen.generate_carrier(seed=2)
        assert not torch.allclose(p1, p2)

    def test_forward_method(self) -> None:
        gen = FourierPatternGenerator(latent_shape=(3, 16, 16), message_bits=8)
        z = torch.randn(3, 16, 16)
        msg = torch.randint(0, 2, (8,)).float()
        out = gen(z, msg, seed=42)
        assert out.shape == z.shape


class TestUserIdToBits:
    def test_deterministic(self) -> None:
        bits1 = _user_id_to_bits("user-42", 48)
        bits2 = _user_id_to_bits("user-42", 48)
        assert torch.equal(bits1, bits2)

    def test_different_ids_different_bits(self) -> None:
        bits1 = _user_id_to_bits("user-42", 48)
        bits2 = _user_id_to_bits("user-99", 48)
        assert not torch.equal(bits1, bits2)

    def test_correct_length(self) -> None:
        bits = _user_id_to_bits("test", 48)
        assert bits.shape == (48,)
        assert set(bits.unique().tolist()).issubset({0.0, 1.0})


class TestLatentWatermarkEmbedder:
    @pytest.fixture()
    def embedder(self) -> LatentWatermarkEmbedder:
        cfg = LatentWatermarkConfig(latent_channels=3, latent_size=16, message_bits=8)
        return LatentWatermarkEmbedder(cfg)

    def test_output_shape(self, embedder: LatentWatermarkEmbedder) -> None:
        z = torch.randn(2, 3, 16, 16)
        z_w, info = embedder.embed(z, secret_key="test-key", user_id="user-1")
        assert z_w.shape == z.shape

    def test_returns_info(self, embedder: LatentWatermarkEmbedder) -> None:
        z = torch.randn(1, 3, 16, 16)
        z_w, info = embedder.embed(z, secret_key="test-key", user_id="user-1")
        assert "pattern" in info
        assert "seed" in info
        assert "alpha" in info
        assert "snr" in info
        assert "user_bits" in info

    def test_approximately_gaussian(self, embedder: LatentWatermarkEmbedder) -> None:
        z = torch.randn(1, 3, 16, 16)
        z_w, _ = embedder.embed(z, secret_key="key", user_id="user-1")
        result = ks_test_gaussianity(z_w)
        assert isinstance(result, dict)
        assert "ks_statistic" in result
        assert result["ks_statistic"] < 0.5  # loose threshold

    def test_forward_returns_tensor(self, embedder: LatentWatermarkEmbedder) -> None:
        z = torch.randn(1, 3, 16, 16)
        z_w = embedder(z, secret_key="key", user_id="u1")
        assert z_w.shape == z.shape

    def test_single_input(self, embedder: LatentWatermarkEmbedder) -> None:
        z = torch.randn(3, 16, 16)
        z_w, info = embedder.embed(z, secret_key="key", user_id="u1")
        assert z_w.shape == (3, 16, 16)


class TestLatentWatermarkDetector:
    @pytest.fixture()
    def detector(self) -> LatentWatermarkDetector:
        cfg = LatentWatermarkConfig(latent_channels=3, latent_size=16, message_bits=8)
        return LatentWatermarkDetector(cfg)

    def test_extract_output_keys(self, detector: LatentWatermarkDetector) -> None:
        z = torch.randn(2, 3, 16, 16)
        result = detector.extract(z, secret_key="test-key")
        for key in ("logits", "predicted_bits", "confidence", "is_watermarked"):
            assert key in result, f"Missing key: {key}"

    def test_logits_shape(self, detector: LatentWatermarkDetector) -> None:
        z = torch.randn(1, 3, 16, 16)
        result = detector.extract(z, secret_key="key")
        assert result["logits"].shape == (1, 8)

    def test_confidence_range(self, detector: LatentWatermarkDetector) -> None:
        z = torch.randn(2, 3, 16, 16)
        result = detector.extract(z, secret_key="key")
        assert result["confidence"].min() >= 0.0
        assert result["confidence"].max() <= 1.0

    def test_forward_returns_bits(self, detector: LatentWatermarkDetector) -> None:
        z = torch.randn(1, 3, 16, 16)
        bits = detector(z, secret_key="test")
        assert bits.shape == (1, 8)
        assert set(bits.unique().tolist()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Pixel embedding
# ---------------------------------------------------------------------------

from backend.ml.pixel_embed import (
    PixelWatermarkConfig,
    PixelWatermarkEmbedder,
    PixelWatermarkExtractor,
    PixelWatermarkLoss,
)


class TestPixelWatermarkEmbedder:
    @pytest.fixture()
    def embedder(self) -> PixelWatermarkEmbedder:
        cfg = PixelWatermarkConfig(message_bits=8, num_blocks=2, hidden_channels=16)
        return PixelWatermarkEmbedder(cfg)

    def test_output_shape(self, embedder: PixelWatermarkEmbedder) -> None:
        img = torch.rand(1, 3, 32, 32)
        msg = torch.randint(0, 2, (1, 8)).float()
        out = embedder(img, msg)
        assert out.shape == img.shape

    def test_output_range(self, embedder: PixelWatermarkEmbedder) -> None:
        img = torch.rand(1, 3, 32, 32)
        msg = torch.randint(0, 2, (1, 8)).float()
        out = embedder(img, msg)
        assert out.min() >= -0.1  # small undershoot OK with learnable strength
        assert out.max() <= 1.1


class TestPixelWatermarkExtractor:
    @pytest.fixture()
    def extractor(self) -> PixelWatermarkExtractor:
        cfg = PixelWatermarkConfig(message_bits=8, num_blocks=2, hidden_channels=16)
        return PixelWatermarkExtractor(cfg)

    def test_extract_blind_keys(self, extractor: PixelWatermarkExtractor) -> None:
        img = torch.rand(2, 3, 32, 32)
        result = extractor.extract_blind(img)
        for key in ("logits", "predicted_bits", "confidence", "is_watermarked"):
            assert key in result

    def test_extract_alias(self, extractor: PixelWatermarkExtractor) -> None:
        """extract() should be an alias for extract_blind()."""
        img = torch.rand(1, 3, 32, 32)
        result = extractor.extract(img)
        for key in ("logits", "predicted_bits", "confidence", "is_watermarked"):
            assert key in result

    def test_logits_shape(self, extractor: PixelWatermarkExtractor) -> None:
        img = torch.rand(1, 3, 32, 32)
        result = extractor.extract_blind(img)
        assert result["logits"].shape == (1, 8)

    def test_forward_returns_logits(self, extractor: PixelWatermarkExtractor) -> None:
        img = torch.rand(1, 3, 32, 32)
        logits = extractor(img)
        assert logits.shape == (1, 8)


class TestPixelWatermarkLoss:
    def test_loss_output_keys(self) -> None:
        loss_fn = PixelWatermarkLoss()
        img = torch.rand(1, 3, 32, 32)
        wm = torch.rand(1, 3, 32, 32)
        logits = torch.randn(1, 8)
        msg = torch.randint(0, 2, (1, 8)).float()
        result = loss_fn(img, wm, logits, msg)
        assert "loss" in result
        assert "bit_acc" in result

    def test_perfect_bits(self) -> None:
        loss_fn = PixelWatermarkLoss()
        img = torch.rand(1, 3, 32, 32)
        msg = torch.ones(1, 8)
        logits = torch.ones(1, 8) * 10.0  # strongly positive → predicted 1
        result = loss_fn(img, img, logits, msg)
        assert result["bit_acc"].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integer Invertible Watermark Network
# ---------------------------------------------------------------------------

from backend.ml.iiwn import IIWNLoss, IntegerInvertibleWatermarkNetwork


class TestIIWN:
    @pytest.fixture()
    def iiwn(self) -> IntegerInvertibleWatermarkNetwork:
        return IntegerInvertibleWatermarkNetwork(
            num_blocks=2, hidden_channels=16, message_bits=8, msg_embed_dim=16,
        )

    def test_embed_shape(self, iiwn: IntegerInvertibleWatermarkNetwork) -> None:
        img = torch.rand(1, 3, 32, 32)
        msg = torch.randint(0, 2, (1, 8)).float()
        wm = iiwn.embed(img, msg)
        assert wm.shape == img.shape

    def test_embed_range(self, iiwn: IntegerInvertibleWatermarkNetwork) -> None:
        img = torch.rand(1, 3, 32, 32)
        msg = torch.randint(0, 2, (1, 8)).float()
        wm = iiwn.embed(img, msg)
        assert wm.min() >= -0.5
        assert wm.max() <= 1.5

    def test_forward_alias(self, iiwn: IntegerInvertibleWatermarkNetwork) -> None:
        img = torch.rand(1, 3, 32, 32)
        msg = torch.randint(0, 2, (1, 8)).float()
        emb = iiwn.embed(img, msg)
        fwd = iiwn(img, msg)
        assert torch.allclose(emb, fwd)

    def test_extract_blind(self, iiwn: IntegerInvertibleWatermarkNetwork) -> None:
        img = torch.rand(1, 3, 32, 32)
        result = iiwn.extract(img)
        for key in ("logits", "predicted_bits", "confidence", "is_watermarked"):
            assert key in result
        assert result["logits"].shape == (1, 8)
        assert result["predicted_bits"].shape == (1, 8)

    def test_extract_confidence_range(self, iiwn: IntegerInvertibleWatermarkNetwork) -> None:
        img = torch.rand(2, 3, 32, 32)
        result = iiwn.extract(img)
        assert result["confidence"].min() >= 0.0
        assert result["confidence"].max() <= 1.0

    def test_gradient_through_embed(self, iiwn: IntegerInvertibleWatermarkNetwork) -> None:
        img = torch.rand(1, 3, 32, 32, requires_grad=True)
        msg = torch.randint(0, 2, (1, 8)).float()
        wm = iiwn.embed(img, msg)
        wm.sum().backward()
        assert img.grad is not None


class TestIIWNLoss:
    def test_loss_output_keys(self) -> None:
        loss_fn = IIWNLoss()
        img = torch.rand(1, 3, 32, 32)
        wm = torch.rand(1, 3, 32, 32)
        result = loss_fn(img, wm)
        assert "loss" in result
        assert "psnr" in result

    def test_loss_with_bits(self) -> None:
        loss_fn = IIWNLoss()
        img = torch.rand(1, 3, 32, 32)
        wm = torch.rand(1, 3, 32, 32)
        logits = torch.randn(1, 8)
        payload = torch.randint(0, 2, (1, 8)).float()
        result = loss_fn(img, wm, logits=logits, payload=payload)
        assert "bit_bce" in result
        assert "bit_acc" in result


# ---------------------------------------------------------------------------
# Attention Decoding Network
# ---------------------------------------------------------------------------

from backend.ml.adn import ADNLoss, AttentionDecodingNetwork


class TestADN:
    @pytest.fixture()
    def adn(self) -> AttentionDecodingNetwork:
        return AttentionDecodingNetwork(feat_dim=32, num_heads=2, num_layers=1, num_res_blocks=1, stride=4)

    def test_output_shapes(self, adn: AttentionDecodingNetwork) -> None:
        img = torch.rand(1, 3, 32, 32)
        attended, mask = adn(img)
        assert attended.shape == img.shape
        assert mask.shape == (1, 1, 32, 32)

    def test_mask_range(self, adn: AttentionDecodingNetwork) -> None:
        img = torch.rand(1, 3, 32, 32)
        _, mask = adn(img)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_gradient_flow(self, adn: AttentionDecodingNetwork) -> None:
        img = torch.rand(1, 3, 32, 32, requires_grad=True)
        attended, mask = adn(img)
        loss = attended.sum() + mask.sum()
        loss.backward()
        assert img.grad is not None


class TestADNLoss:
    def test_loss_output(self) -> None:
        loss_fn = ADNLoss()
        mask = torch.rand(1, 1, 16, 16)
        result = loss_fn(mask)
        assert "loss" in result
        assert result["loss"].ndim == 0


# ---------------------------------------------------------------------------
# Error correction (crypto)
# ---------------------------------------------------------------------------

from backend.ml.crypto import (
    HammingInterleavedCodec,
    bits_to_hex,
    decode_payload,
    encode_payload,
    hex_to_bits,
)


class TestCrypto:
    @pytest.fixture(autouse=True)
    def _check_reedsolo(self) -> None:
        pytest.importorskip("reedsolo", reason="reedsolo not installed")

    def test_round_trip_no_errors(self) -> None:
        import numpy as np
        original = np.random.randint(0, 2, size=(48,)).astype(np.uint8)
        encoded = encode_payload(original)
        decoded = decode_payload(encoded)
        assert np.array_equal(original, decoded)

    def test_encoded_length(self) -> None:
        import numpy as np
        original = np.random.randint(0, 2, size=(48,)).astype(np.uint8)
        encoded = encode_payload(original)
        assert encoded.shape[0] == 256

    def test_error_correction(self) -> None:
        """Flip a few bits and verify correction."""
        import numpy as np
        np.random.seed(42)
        original = np.random.randint(0, 2, size=(48,)).astype(np.uint8)
        encoded = encode_payload(original)
        # Flip 5 random bits
        flip_idx = np.random.choice(256, size=5, replace=False)
        corrupted = encoded.copy()
        corrupted[flip_idx] = 1 - corrupted[flip_idx]
        decoded = decode_payload(corrupted)
        assert np.array_equal(original, decoded)

    def test_hex_roundtrip(self) -> None:
        import numpy as np
        # 48 bits → 12 hex chars
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 6, dtype=np.uint8)  # 48 bits
        hex_str = bits_to_hex(bits)
        assert len(hex_str) == 12
        recovered = hex_to_bits(hex_str)
        assert np.array_equal(bits, recovered)

    def test_hamming_codec(self) -> None:
        import numpy as np
        codec = HammingInterleavedCodec()
        data = np.random.randint(0, 2, size=(48,)).astype(np.uint8)
        encoded = codec.encode(data)
        decoded = codec.decode(encoded)
        assert np.array_equal(data, decoded)


# ---------------------------------------------------------------------------
# Discrepancy detector
# ---------------------------------------------------------------------------

from backend.ml.discrepancy import (
    AttackType,
    DiscrepancyLoss,
    LearnedDiscrepancy,
    RuleBasedDiscrepancy,
    build_feature_vector,
)


def _make_result(conf: float, bits: list[int]) -> dict[str, torch.Tensor]:
    """Helper to build a fake extractor result dict."""
    N = len(bits)
    bits_t = torch.tensor([bits], dtype=torch.float32)
    logits = (bits_t * 2 - 1) * conf * 5  # scale logits by confidence
    return {
        "logits": logits,
        "predicted_bits": bits_t,
        "confidence": torch.tensor([conf]),
        "is_watermarked": torch.tensor([conf > 0.5]),
    }


class TestAttackTypeEnum:
    def test_diffusion_exists(self) -> None:
        assert hasattr(AttackType, "DIFFUSION")
        assert AttackType.DIFFUSION.value == "diffusion"

    def test_all_members(self) -> None:
        expected = {"CLEAN", "JPEG", "CROP", "D2RA", "DAWN", "DIFFUSION"}
        actual = {m.name for m in AttackType}
        assert actual == expected

    def test_has_six_members(self) -> None:
        assert len(AttackType) == 6


class TestBuildFeatureVector:
    def test_shape(self) -> None:
        lat = _make_result(0.9, [1, 0, 1, 0])
        pix = _make_result(0.8, [1, 0, 1, 0])
        features = build_feature_vector(lat, pix)
        assert features.shape == (1, 7)


class TestRuleBasedDiscrepancy:
    def test_clean_detection(self) -> None:
        detector = RuleBasedDiscrepancy()
        bits = [1, 0, 1] * 16  # 48 bits
        lat = _make_result(0.9, bits)
        pix = _make_result(0.85, bits)
        result = detector.classify(lat, pix)
        assert result["attack_type"][0] == AttackType.CLEAN

    def test_d2ra_detection(self) -> None:
        detector = RuleBasedDiscrepancy()
        bits = [1, 0, 1] * 16
        lat = _make_result(0.9, bits)
        pix = _make_result(0.2, [0] * 48)  # pixel scrubbed
        result = detector.classify(lat, pix)
        assert result["attack_type"][0] == AttackType.D2RA

    def test_crop_detection(self) -> None:
        detector = RuleBasedDiscrepancy()
        lat = _make_result(0.2, [0] * 48)
        pix = _make_result(0.2, [0] * 48)
        result = detector.classify(lat, pix)
        assert result["attack_type"][0] == AttackType.CROP

    def test_confidence_range(self) -> None:
        detector = RuleBasedDiscrepancy()
        bits = [1, 0, 1] * 16
        lat = _make_result(0.9, bits)
        pix = _make_result(0.85, bits)
        result = detector.classify(lat, pix)
        assert result["confidence"].min() >= 0.0
        assert result["confidence"].max() <= 1.0


class TestLearnedDiscrepancy:
    def test_forward_shape(self) -> None:
        clf = LearnedDiscrepancy()
        lat = _make_result(0.9, [1, 0] * 24)
        pix = _make_result(0.8, [1, 0] * 24)
        logits = clf(lat, pix)
        assert logits.shape == (1, LearnedDiscrepancy.NUM_CLASSES)

    def test_num_classes_is_six(self) -> None:
        assert LearnedDiscrepancy.NUM_CLASSES == 6

    def test_predict_keys(self) -> None:
        clf = LearnedDiscrepancy()
        lat = _make_result(0.9, [1, 0] * 24)
        pix = _make_result(0.8, [1, 0] * 24)
        result = clf.predict(lat, pix)
        assert "attack_type" in result
        assert "probabilities" in result
        assert "confidence" in result
        assert isinstance(result["attack_type"][0], AttackType)

    def test_gradient_flow(self) -> None:
        clf = LearnedDiscrepancy()
        lat = _make_result(0.9, [1, 0] * 24)
        pix = _make_result(0.8, [1, 0] * 24)
        logits = clf(lat, pix)
        loss = logits.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in clf.parameters())
        assert has_grad


class TestDiscrepancyLoss:
    def test_loss_output(self) -> None:
        loss_fn = DiscrepancyLoss()
        logits = torch.randn(4, LearnedDiscrepancy.NUM_CLASSES)
        targets = torch.randint(0, LearnedDiscrepancy.NUM_CLASSES, (4,))
        result = loss_fn(logits, targets)
        assert "loss" in result
        assert "accuracy" in result
        assert result["loss"].ndim == 0

    def test_perfect_predictions(self) -> None:
        loss_fn = DiscrepancyLoss(label_smoothing=0.0)
        logits = torch.zeros(3, LearnedDiscrepancy.NUM_CLASSES)
        targets = torch.tensor([0, 1, 2])
        logits[0, 0] = 100.0
        logits[1, 1] = 100.0
        logits[2, 2] = 100.0
        result = loss_fn(logits, targets)
        assert result["accuracy"].item() == pytest.approx(1.0)
