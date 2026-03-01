"""Phase 5 — Content-Dependent Cryptographic Binding (MetaSeal) tests.

Covers:
  - perceptual_hash.py: DINOv2PerceptualHasher (stub mode), sha256_tensor_hash
  - ecdsa_signer.py: ECDSAKeyPair, ECDSASigner, ECDSAVerifier, bit↔byte utils
  - forgery_detector.py: ForgeryDetector, MetaSealPipeline
"""

from __future__ import annotations

import pytest
import torch

# =====================================================================
# Helpers
# =====================================================================

B, C, H, W = 2, 3, 64, 64
HASH_DIM = 512


def _rand_image(b: int = B, h: int = H, w: int = W) -> torch.Tensor:
    """Random RGB tensor in [0, 1]."""
    return torch.rand(b, C, h, w)


# =====================================================================
# perceptual_hash.py
# =====================================================================

from backend.ml.perceptual_hash import (
    DINOv2PerceptualHasher,
    PerceptualHashConfig,
    sha256_tensor_hash,
)


class TestPerceptualHashConfig:
    def test_defaults(self) -> None:
        cfg = PerceptualHashConfig()
        assert cfg.backbone == "dinov2_vitb14"
        assert cfg.hash_dim == 512
        assert cfg.backbone_dim == 768
        assert cfg.input_size == 224

    def test_custom(self) -> None:
        cfg = PerceptualHashConfig(hash_dim=256, input_size=128)
        assert cfg.hash_dim == 256
        assert cfg.input_size == 128


class TestDINOv2PerceptualHasher:
    """Tests use stub backbone (use_torch_hub=False) for speed."""

    @pytest.fixture()
    def hasher(self) -> DINOv2PerceptualHasher:
        cfg = PerceptualHashConfig(use_torch_hub=False)
        return DINOv2PerceptualHasher(cfg)

    def test_output_shape(self, hasher: DINOv2PerceptualHasher) -> None:
        img = _rand_image(2)
        h = hasher.hash(img)
        assert h.shape == (2, HASH_DIM)

    def test_l2_normalised(self, hasher: DINOv2PerceptualHasher) -> None:
        h = hasher.hash(_rand_image(3))
        norms = h.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    def test_forward_alias(self, hasher: DINOv2PerceptualHasher) -> None:
        img = _rand_image(1)
        h1 = hasher.hash(img)
        h2 = hasher(img)
        assert torch.allclose(h1, h2)

    def test_deterministic(self, hasher: DINOv2PerceptualHasher) -> None:
        torch.manual_seed(42)
        img = _rand_image(1)
        h1 = hasher.hash(img)
        h2 = hasher.hash(img)
        assert torch.allclose(h1, h2)

    def test_different_images_differ(self, hasher: DINOv2PerceptualHasher) -> None:
        torch.manual_seed(1)
        img_a = _rand_image(1)
        torch.manual_seed(2)
        img_b = _rand_image(1)
        ha = hasher.hash(img_a)
        hb = hasher.hash(img_b)
        assert not torch.allclose(ha, hb, atol=1e-3)

    def test_compute_similarity(self, hasher: DINOv2PerceptualHasher) -> None:
        h = hasher.hash(_rand_image(2))
        sim_self = hasher.compute_similarity(h, h)
        assert torch.allclose(sim_self, torch.ones(2), atol=1e-5)

    def test_are_similar_self(self, hasher: DINOv2PerceptualHasher) -> None:
        h = hasher.hash(_rand_image(2))
        assert hasher.are_similar(h, h).all()

    def test_various_input_sizes(self, hasher: DINOv2PerceptualHasher) -> None:
        """Should handle different input resolutions via resize."""
        for size in [(32, 32), (128, 128), (224, 224), (300, 200)]:
            img = torch.rand(1, 3, *size)
            h = hasher.hash(img)
            assert h.shape == (1, HASH_DIM)

    def test_backbone_frozen(self, hasher: DINOv2PerceptualHasher) -> None:
        """Backbone parameters should not require gradients."""
        for p in hasher.backbone.parameters():
            assert not p.requires_grad

    def test_projection_trainable(self, hasher: DINOv2PerceptualHasher) -> None:
        """Projection head parameters should require gradients."""
        for p in hasher.projection.parameters():
            assert p.requires_grad


class TestSha256TensorHash:
    def test_returns_32_bytes(self) -> None:
        t = torch.randn(10)
        digest = sha256_tensor_hash(t)
        assert isinstance(digest, bytes)
        assert len(digest) == 32

    def test_deterministic(self) -> None:
        t = torch.randn(5, 5)
        d1 = sha256_tensor_hash(t)
        d2 = sha256_tensor_hash(t)
        assert d1 == d2

    def test_different_tensors_differ(self) -> None:
        t1 = torch.ones(4)
        t2 = torch.zeros(4)
        assert sha256_tensor_hash(t1) != sha256_tensor_hash(t2)


# =====================================================================
# ecdsa_signer.py
# =====================================================================

from backend.ml.ecdsa_signer import (
    ECDSAConfig,
    ECDSAKeyPair,
    ECDSASigner,
    ECDSAVerifier,
    _embedding_to_digest,
    bits_to_signature,
    hex_to_signature,
    signature_to_bits,
    signature_to_hex,
)


class TestECDSAConfig:
    def test_defaults(self) -> None:
        cfg = ECDSAConfig()
        assert cfg.curve_name == "secp256k1"
        assert cfg.signature_bytes == 64


class TestECDSAKeyPair:
    def test_generate(self) -> None:
        kp = ECDSAKeyPair()
        assert kp.private_key is not None
        assert kp.public_key is not None

    def test_pem_roundtrip(self) -> None:
        kp = ECDSAKeyPair()
        pem = kp.private_pem()
        kp2 = ECDSAKeyPair.from_private_pem(pem)
        # Same key should produce same public PEM
        assert kp.public_pem() == kp2.public_pem()

    def test_public_pem_export(self) -> None:
        kp = ECDSAKeyPair()
        pub_pem = kp.public_pem()
        assert b"PUBLIC KEY" in pub_pem

    def test_private_pem_export(self) -> None:
        kp = ECDSAKeyPair()
        priv_pem = kp.private_pem()
        assert b"PRIVATE KEY" in priv_pem

    def test_custom_curve(self) -> None:
        cfg = ECDSAConfig(curve_name="secp256r1")
        kp = ECDSAKeyPair(config=cfg)
        assert kp.private_key is not None

    def test_unsupported_curve_raises(self) -> None:
        cfg = ECDSAConfig(curve_name="invalid_curve")
        with pytest.raises(ValueError, match="Unsupported curve"):
            ECDSAKeyPair(config=cfg)

    def test_from_public_pem(self) -> None:
        kp = ECDSAKeyPair()
        pub_pem = kp.public_pem()
        pub_key = ECDSAKeyPair.from_public_pem(pub_pem)
        assert pub_key is not None


class TestECDSASignerVerifier:
    @pytest.fixture()
    def key_pair(self) -> ECDSAKeyPair:
        return ECDSAKeyPair()

    @pytest.fixture()
    def signer(self, key_pair: ECDSAKeyPair) -> ECDSASigner:
        return ECDSASigner(key_pair)

    @pytest.fixture()
    def verifier(self, key_pair: ECDSAKeyPair) -> ECDSAVerifier:
        return ECDSAVerifier(key_pair.public_key)

    def test_sign_embedding_returns_64_bytes(
        self, signer: ECDSASigner
    ) -> None:
        emb = torch.randn(512)
        sig = signer.sign_embedding(emb)
        assert isinstance(sig, bytes)
        assert len(sig) == 64

    def test_verify_valid_signature(
        self, signer: ECDSASigner, verifier: ECDSAVerifier
    ) -> None:
        emb = torch.randn(512)
        sig = signer.sign_embedding(emb)
        assert verifier.verify_embedding(emb, sig) is True

    def test_verify_wrong_embedding_fails(
        self, signer: ECDSASigner, verifier: ECDSAVerifier
    ) -> None:
        emb = torch.randn(512)
        sig = signer.sign_embedding(emb)
        wrong_emb = torch.randn(512)
        assert verifier.verify_embedding(wrong_emb, sig) is False

    def test_verify_tampered_signature_fails(
        self, signer: ECDSASigner, verifier: ECDSAVerifier
    ) -> None:
        emb = torch.randn(512)
        sig = signer.sign_embedding(emb)
        # Flip a byte
        tampered = bytearray(sig)
        tampered[0] ^= 0xFF
        assert verifier.verify_embedding(emb, bytes(tampered)) is False

    def test_verify_wrong_key_fails(self, signer: ECDSASigner) -> None:
        emb = torch.randn(512)
        sig = signer.sign_embedding(emb)
        # Different key pair
        other_kp = ECDSAKeyPair()
        other_verifier = ECDSAVerifier(other_kp.public_key)
        assert other_verifier.verify_embedding(emb, sig) is False

    def test_sign_digest_directly(self, signer: ECDSASigner, verifier: ECDSAVerifier) -> None:
        import hashlib
        digest = hashlib.sha256(b"test data").digest()
        sig = signer.sign_digest(digest)
        assert len(sig) == 64
        assert verifier.verify_digest(digest, sig) is True

    def test_sign_embedding_to_bits(self, signer: ECDSASigner) -> None:
        emb = torch.randn(512)
        bits = signer.sign_embedding_to_bits(emb)
        assert bits.shape == (512,)
        assert set(bits.unique().tolist()).issubset({0.0, 1.0})

    def test_verify_embedding_bits(
        self, signer: ECDSASigner, verifier: ECDSAVerifier
    ) -> None:
        emb = torch.randn(512)
        bits = signer.sign_embedding_to_bits(emb)
        assert verifier.verify_embedding_bits(emb, bits) is True

    def test_batched_embedding(
        self, signer: ECDSASigner, verifier: ECDSAVerifier
    ) -> None:
        """2D embedding (1, 512) should also work."""
        emb = torch.randn(1, 512)
        sig = signer.sign_embedding(emb)
        assert verifier.verify_embedding(emb, sig) is True

    def test_short_signature_fails(self, verifier: ECDSAVerifier) -> None:
        emb = torch.randn(512)
        assert verifier.verify_embedding(emb, b"\x00" * 32) is False


class TestBitByteConversion:
    def test_roundtrip(self) -> None:
        sig = bytes(range(64))
        bits = signature_to_bits(sig)
        assert bits.shape == (512,)
        recovered = bits_to_signature(bits)
        assert recovered == sig

    def test_bits_values(self) -> None:
        sig = b"\xff" + b"\x00" * 63
        bits = signature_to_bits(sig)
        assert bits[:8].tolist() == [1, 1, 1, 1, 1, 1, 1, 1]
        assert bits[8:16].tolist() == [0, 0, 0, 0, 0, 0, 0, 0]

    def test_hex_roundtrip(self) -> None:
        sig = bytes(range(64))
        hex_str = signature_to_hex(sig)
        assert hex_to_signature(hex_str) == sig

    def test_wrong_bit_length_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected 512"):
            bits_to_signature(torch.zeros(256))


class TestEmbeddingToDigest:
    def test_returns_32_bytes(self) -> None:
        emb = torch.randn(512)
        digest = _embedding_to_digest(emb)
        assert len(digest) == 32

    def test_deterministic(self) -> None:
        emb = torch.randn(512)
        d1 = _embedding_to_digest(emb)
        d2 = _embedding_to_digest(emb)
        assert d1 == d2

    def test_normalises_input(self) -> None:
        """Same direction, different magnitude → same digest."""
        emb = torch.randn(512)
        d1 = _embedding_to_digest(emb)
        d2 = _embedding_to_digest(emb * 3.0)
        assert d1 == d2


# =====================================================================
# forgery_detector.py
# =====================================================================

from backend.ml.forgery_detector import (
    ForgeryDetector,
    ForgeryDetectorConfig,
    ForgeryResult,
    MetaSealPipeline,
)


class TestForgeryDetector:
    @pytest.fixture()
    def setup(self):
        hash_cfg = PerceptualHashConfig(use_torch_hub=False)
        det_cfg = ForgeryDetectorConfig(hash_config=hash_cfg)
        detector = ForgeryDetector(config=det_cfg)
        key_pair = ECDSAKeyPair()
        signer = ECDSASigner(key_pair)
        return detector, key_pair, signer

    def test_genuine_image_passes(self, setup) -> None:
        detector, key_pair, signer = setup
        img = _rand_image(1)
        # Hash and sign
        h = detector.hasher.hash(img)[0]
        sig = signer.sign_embedding(h)
        # Verify same image
        result = detector.detect(img, sig, key_pair.public_key)
        assert isinstance(result, ForgeryResult)
        assert result.ecdsa_valid is True
        assert result.forgery_detected is False

    def test_transplanted_image_detected(self, setup) -> None:
        detector, key_pair, signer = setup
        torch.manual_seed(10)
        img_original = _rand_image(1)
        torch.manual_seed(20)
        img_forgery = _rand_image(1)
        # Hash and sign ORIGINAL
        h = detector.hasher.hash(img_original)[0]
        sig = signer.sign_embedding(h)
        # Verify against DIFFERENT image
        result = detector.detect(img_forgery, sig, key_pair.public_key)
        assert result.ecdsa_valid is False
        assert result.forgery_detected is True

    def test_detect_from_bits(self, setup) -> None:
        detector, key_pair, signer = setup
        img = _rand_image(1)
        h = detector.hasher.hash(img)[0]
        sig_bits = signer.sign_embedding_to_bits(h)
        result = detector.detect_from_bits(img, sig_bits, key_pair.public_key)
        assert result.ecdsa_valid is True
        assert result.forgery_detected is False

    def test_forward_returns_dict(self, setup) -> None:
        detector, key_pair, signer = setup
        img = _rand_image(1)
        h = detector.hasher.hash(img)[0]
        sig_bits = signer.sign_embedding_to_bits(h)
        out = detector(img, sig_bits, key_pair.public_key)
        assert "forgery_detected" in out
        assert "ecdsa_valid" in out
        assert "confidence" in out
        assert "recomputed_hash" in out

    def test_reference_hash_similarity(self, setup) -> None:
        detector, key_pair, signer = setup
        img = _rand_image(1)
        h = detector.hasher.hash(img)[0]
        sig = signer.sign_embedding(h)
        result = detector.detect(img, sig, key_pair.public_key, reference_hash=h)
        assert result.content_similarity > 0.99  # same image, same hash

    def test_3d_image_input(self, setup) -> None:
        """(3, H, W) without batch dim should work."""
        detector, key_pair, signer = setup
        img = torch.rand(3, 64, 64)
        h = detector.hasher.hash(img.unsqueeze(0))[0]
        sig = signer.sign_embedding(h)
        result = detector.detect(img, sig, key_pair.public_key)
        assert result.ecdsa_valid is True

    def test_soft_mode(self) -> None:
        hash_cfg = PerceptualHashConfig(use_torch_hub=False)
        det_cfg = ForgeryDetectorConfig(hash_config=hash_cfg, strict_mode=False)
        detector = ForgeryDetector(config=det_cfg)
        kp = ECDSAKeyPair()
        signer = ECDSASigner(kp)
        img = _rand_image(1)
        h = detector.hasher.hash(img)[0]
        sig = signer.sign_embedding(h)
        result = detector.detect(img, sig, kp.public_key)
        # In soft mode, valid sig → low forgery score
        assert result.forgery_detected is False


class TestMetaSealPipeline:
    @pytest.fixture()
    def pipeline(self) -> MetaSealPipeline:
        hash_cfg = PerceptualHashConfig(use_torch_hub=False)
        hasher = DINOv2PerceptualHasher(hash_cfg)
        key_pair = ECDSAKeyPair()
        det_cfg = ForgeryDetectorConfig(hash_config=hash_cfg)
        return MetaSealPipeline(hasher, key_pair, config=det_cfg)

    def test_sign_image(self, pipeline: MetaSealPipeline) -> None:
        img = _rand_image(1)
        result = pipeline.sign_image(img)
        assert result["hash"].shape == (HASH_DIM,)
        assert len(result["signature"]) == 64
        assert result["signature_bits"].shape == (512,)
        assert isinstance(result["signature_hex"], str)
        assert len(result["signature_hex"]) == 128

    def test_sign_and_verify_roundtrip(self, pipeline: MetaSealPipeline) -> None:
        img = _rand_image(1)
        signed = pipeline.sign_image(img)
        result = pipeline.verify_image(img, signed["signature"])
        assert result.ecdsa_valid is True
        assert result.forgery_detected is False

    def test_sign_and_verify_with_reference(self, pipeline: MetaSealPipeline) -> None:
        img = _rand_image(1)
        signed = pipeline.sign_image(img)
        result = pipeline.verify_image(
            img, signed["signature"], reference_hash=signed["hash"]
        )
        assert result.content_similarity > 0.99

    def test_verify_from_bits(self, pipeline: MetaSealPipeline) -> None:
        img = _rand_image(1)
        signed = pipeline.sign_image(img)
        result = pipeline.verify_image_from_bits(img, signed["signature_bits"])
        assert result.ecdsa_valid is True

    def test_forgery_detected_different_image(self, pipeline: MetaSealPipeline) -> None:
        torch.manual_seed(100)
        img_a = _rand_image(1)
        signed = pipeline.sign_image(img_a)
        torch.manual_seed(200)
        img_b = _rand_image(1)
        result = pipeline.verify_image(img_b, signed["signature"])
        assert result.forgery_detected is True

    def test_3d_image_sign(self, pipeline: MetaSealPipeline) -> None:
        """(3, H, W) without batch dim."""
        img = torch.rand(3, 64, 64)
        signed = pipeline.sign_image(img)
        assert signed["hash"].shape == (HASH_DIM,)
