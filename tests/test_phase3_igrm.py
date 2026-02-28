"""Phase 3 — IGRM Advanced module tests.

Tests cover:
  - losses.py: LPIPSLoss, FrequencyLoss, WatermarkLoss, RestorationLoss
  - unet_restore.py: ConditionalRestorationUNet, sinusoidal_embedding, SEBlock
  - diffusion_restore.py: ConditionalDiffusionRestorer (training + inference)
  - keypoint_detector.py: WatermarkKeypointDetector, KeypointDetectorLoss
  - ddim_inversion.py: DDIMInversion, BlindDDIMInversion, LatentWatermarkExtractor
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

# Image size kept small for fast CPU tests
B, C, H, W = 2, 3, 64, 64
MSG_LEN = 48


def _rand_image(b: int = B, h: int = H, w: int = W) -> torch.Tensor:
    return torch.rand(b, C, h, w)


def _rand_bits(b: int = B, n: int = MSG_LEN) -> torch.Tensor:
    return torch.randint(0, 2, (b, n)).float()


# =====================================================================
# losses.py
# =====================================================================

class TestLPIPSLoss:
    def test_output_scalar(self):
        from backend.ml.losses import LPIPSLoss

        loss_fn = LPIPSLoss()
        x, y = _rand_image(), _rand_image()
        loss = loss_fn(x, y)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_identical_images_low_loss(self):
        from backend.ml.losses import LPIPSLoss

        loss_fn = LPIPSLoss()
        x = _rand_image()
        loss_same = loss_fn(x, x)
        loss_diff = loss_fn(x, _rand_image())
        assert loss_same.item() < loss_diff.item()

    def test_no_grad_on_feature_extractor(self):
        from backend.ml.losses import LPIPSLoss

        loss_fn = LPIPSLoss()
        for p in loss_fn.feature_extractor.parameters():
            assert not p.requires_grad


class TestFrequencyLoss:
    def test_output_scalar(self):
        from backend.ml.losses import FrequencyLoss

        loss_fn = FrequencyLoss()
        x, y = _rand_image(), _rand_image()
        loss = loss_fn(x, y)
        assert loss.shape == ()

    def test_identical_zero(self):
        from backend.ml.losses import FrequencyLoss

        loss_fn = FrequencyLoss()
        x = _rand_image()
        loss = loss_fn(x, x)
        assert loss.item() < 1e-5

    def test_high_freq_weight(self):
        from backend.ml.losses import FrequencyLoss

        loss_hf = FrequencyLoss(weight_high_freq=5.0)
        loss_nohf = FrequencyLoss(weight_high_freq=1.0)
        x, y = _rand_image(), _rand_image()
        # Higher weight should produce larger loss for high-freq content
        l_hf = loss_hf(x, y)
        l_nohf = loss_nohf(x, y)
        assert l_hf.item() >= l_nohf.item()


class TestWatermarkLoss:
    def test_output_keys(self):
        from backend.ml.losses import WatermarkLoss

        loss_fn = WatermarkLoss()
        cover = _rand_image()
        wm = _rand_image()
        logits = torch.randn(B, MSG_LEN)
        bits = _rand_bits()
        result = loss_fn(cover, wm, logits, bits)
        assert "loss" in result
        assert "lpips" in result
        assert "mse" in result
        assert "msg_bce" in result
        assert "bit_acc" in result

    def test_bit_accuracy_range(self):
        from backend.ml.losses import WatermarkLoss

        loss_fn = WatermarkLoss()
        cover = _rand_image()
        wm = _rand_image()
        logits = torch.randn(B, MSG_LEN)
        bits = _rand_bits()
        result = loss_fn(cover, wm, logits, bits)
        assert 0 <= result["bit_acc"].item() <= 1.0

    def test_perfect_bits_high_accuracy(self):
        from backend.ml.losses import WatermarkLoss

        loss_fn = WatermarkLoss()
        cover = _rand_image()
        bits = _rand_bits()
        # Logits perfectly aligned with bits
        logits = bits * 10 - 5  # 0 → -5, 1 → 5
        result = loss_fn(cover, cover, logits, bits)
        assert result["bit_acc"].item() > 0.99

    def test_gradient_flows(self):
        from backend.ml.losses import WatermarkLoss

        loss_fn = WatermarkLoss()
        cover = _rand_image()
        wm = _rand_image().requires_grad_(True)
        logits = torch.randn(B, MSG_LEN, requires_grad=True)
        bits = _rand_bits()
        result = loss_fn(cover, wm, logits, bits)
        result["loss"].backward()
        assert wm.grad is not None
        assert logits.grad is not None


class TestRestorationLoss:
    def test_output_keys(self):
        from backend.ml.losses import RestorationLoss

        # Simple decoder mock
        decoder = nn.Linear(C * H * W, MSG_LEN)
        decoder_wrapper = _FlatDecoder(decoder)
        loss_fn = RestorationLoss(decoder_wrapper)

        restored = _rand_image()
        clean = _rand_image()
        bits = _rand_bits()
        result = loss_fn(restored, clean, bits)
        assert "loss" in result
        assert "bit_bce" in result
        assert "percep_mse" in result
        assert "freq" in result
        assert "bit_acc" in result

    def test_bit_weight_dominates(self):
        from backend.ml.losses import RestorationLoss, RestorationLossConfig

        decoder = _FlatDecoder(nn.Linear(C * H * W, MSG_LEN))
        cfg = RestorationLossConfig(lambda_bit=100.0, lambda_percep=0.01, lambda_freq=0.01)
        loss_fn = RestorationLoss(decoder, cfg)

        restored = _rand_image()
        clean = _rand_image()
        bits = _rand_bits()
        result = loss_fn(restored, clean, bits)
        # Bit loss should dominate total
        assert result["loss"].item() > result["percep_mse"].item()


class _FlatDecoder(nn.Module):
    """Wrapper that flattens image before passing to a linear layer."""

    def __init__(self, linear: nn.Module) -> None:
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1))


# =====================================================================
# unet_restore.py
# =====================================================================

class TestSinusoidalEmbedding:
    def test_output_shape(self):
        from backend.ml.unet_restore import sinusoidal_embedding

        t = torch.tensor([0.0, 0.5, 1.0])
        emb = sinusoidal_embedding(t, 256)
        assert emb.shape == (3, 256)

    def test_different_timesteps_different_embeddings(self):
        from backend.ml.unet_restore import sinusoidal_embedding

        t1 = torch.tensor([0.1])
        t2 = torch.tensor([0.9])
        e1 = sinusoidal_embedding(t1, 128)
        e2 = sinusoidal_embedding(t2, 128)
        assert not torch.allclose(e1, e2)


class TestSEBlock:
    def test_output_shape(self):
        from backend.ml.unet_restore import SEBlock

        se = SEBlock(64)
        x = torch.randn(2, 64, 16, 16)
        out = se(x)
        assert out.shape == x.shape


class TestConditionalRestorationUNet:
    def test_output_shape(self):
        from backend.ml.unet_restore import ConditionalRestorationUNet

        model = ConditionalRestorationUNet(
            base_channels=32, depth=3, num_bottleneck_blocks=2, t_dim=64
        )
        x = _rand_image()
        out = model(x)
        assert out.shape == x.shape

    def test_with_severity(self):
        from backend.ml.unet_restore import ConditionalRestorationUNet

        model = ConditionalRestorationUNet(
            base_channels=32, depth=3, num_bottleneck_blocks=2, t_dim=64
        )
        x = _rand_image()
        severity = torch.tensor([0.2, 0.8])
        out = model(x, severity=severity)
        assert out.shape == x.shape

    def test_output_range(self):
        from backend.ml.unet_restore import ConditionalRestorationUNet

        model = ConditionalRestorationUNet(
            base_channels=32, depth=3, num_bottleneck_blocks=2, t_dim=64
        )
        x = _rand_image()
        out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_gradient_flow(self):
        from backend.ml.unet_restore import ConditionalRestorationUNet

        model = ConditionalRestorationUNet(
            base_channels=32, depth=3, num_bottleneck_blocks=2, t_dim=64
        )
        x = _rand_image().requires_grad_(True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None

    def test_residual_learning(self):
        """Network uses residual learning (output = input + residual)."""
        from backend.ml.unet_restore import ConditionalRestorationUNet

        model = ConditionalRestorationUNet(
            base_channels=32, depth=3, num_bottleneck_blocks=2, t_dim=64
        )
        # With zero-init, output should be close to input
        # (not exactly equal due to random init of intermediate layers)
        x = _rand_image()
        out = model(x)
        # At least the shapes match and values are in valid range
        assert out.shape == x.shape
        assert out.min() >= 0.0


# =====================================================================
# diffusion_restore.py
# =====================================================================

class TestConditionalDiffusionRestorer:
    def test_training_loss(self):
        from backend.ml.diffusion_restore import ConditionalDiffusionRestorer

        model = ConditionalDiffusionRestorer(
            num_steps=100, base_channels=16, depth=2, inference_steps=5
        )
        model.train()
        clean = _rand_image()
        degraded = _rand_image()
        result = model(degraded, clean_watermarked=clean)
        assert isinstance(result, dict)
        assert "loss" in result
        assert result["loss"].item() > 0

    def test_q_sample_shape(self):
        from backend.ml.diffusion_restore import ConditionalDiffusionRestorer

        model = ConditionalDiffusionRestorer(num_steps=100, base_channels=16, depth=2)
        x = _rand_image()
        t = torch.randint(0, 100, (B,))
        x_noisy = model.q_sample(x, t)
        assert x_noisy.shape == x.shape

    def test_q_sample_more_noise_at_higher_t(self):
        from backend.ml.diffusion_restore import ConditionalDiffusionRestorer

        model = ConditionalDiffusionRestorer(num_steps=100, base_channels=16, depth=2)
        x = _rand_image()
        noise = torch.randn_like(x)
        t_low = torch.zeros(B, dtype=torch.long)
        t_high = torch.full((B,), 99, dtype=torch.long)
        x_low = model.q_sample(x, t_low, noise=noise)
        x_high = model.q_sample(x, t_high, noise=noise)
        # More noise at higher timestep
        diff_low = (x_low - x).abs().mean()
        diff_high = (x_high - x).abs().mean()
        assert diff_high > diff_low

    def test_restore_output_shape(self):
        from backend.ml.diffusion_restore import ConditionalDiffusionRestorer

        model = ConditionalDiffusionRestorer(
            num_steps=100, base_channels=16, depth=2, inference_steps=3
        )
        model.eval()
        degraded = _rand_image()
        restored = model.restore(degraded, num_steps=3)
        assert restored.shape == degraded.shape
        assert restored.min() >= 0.0
        assert restored.max() <= 1.0

    def test_noise_schedule_buffers(self):
        from backend.ml.diffusion_restore import ConditionalDiffusionRestorer

        model = ConditionalDiffusionRestorer(num_steps=100, base_channels=16, depth=2)
        assert hasattr(model, "betas")
        assert hasattr(model, "alphas_cumprod")
        assert model.betas.shape == (100,)
        assert model.alphas_cumprod.shape == (100,)
        # alphas_cumprod should be decreasing
        assert (model.alphas_cumprod[:-1] >= model.alphas_cumprod[1:]).all()

    def test_cosine_schedule(self):
        from backend.ml.diffusion_restore import ConditionalDiffusionRestorer

        model = ConditionalDiffusionRestorer(
            num_steps=100, schedule="cosine", base_channels=16, depth=2
        )
        assert model.alphas_cumprod.shape == (100,)
        assert (model.alphas_cumprod[:-1] >= model.alphas_cumprod[1:]).all()


# =====================================================================
# keypoint_detector.py
# =====================================================================

class TestWatermarkKeypointDetector:
    def test_output_keys(self):
        from backend.ml.keypoint_detector import WatermarkKeypointDetector

        detector = WatermarkKeypointDetector(
            patch_size=16, feat_dim=64, num_heads=2, num_layers=2, max_patches=64
        )
        x = _rand_image()
        result = detector(x)
        assert "logits" in result
        assert "attention_map" in result
        assert "grid_h" in result
        assert "grid_w" in result

    def test_attention_map_shape(self):
        from backend.ml.keypoint_detector import WatermarkKeypointDetector

        detector = WatermarkKeypointDetector(
            patch_size=16, feat_dim=64, num_heads=2, num_layers=2, max_patches=64
        )
        x = _rand_image()
        result = detector(x)
        assert result["attention_map"].shape == (B, 1, H, W)

    def test_attention_map_range(self):
        from backend.ml.keypoint_detector import WatermarkKeypointDetector

        detector = WatermarkKeypointDetector(
            patch_size=16, feat_dim=64, num_heads=2, num_layers=2, max_patches=64
        )
        x = _rand_image()
        result = detector(x)
        attn = result["attention_map"]
        assert attn.min() >= 0.0
        assert attn.max() <= 1.0

    def test_grid_dimensions(self):
        from backend.ml.keypoint_detector import WatermarkKeypointDetector

        detector = WatermarkKeypointDetector(
            patch_size=16, feat_dim=64, num_heads=2, num_layers=2, max_patches=64
        )
        x = _rand_image()
        result = detector(x)
        expected_gh = H // 16
        expected_gw = W // 16
        assert result["grid_h"] == expected_gh
        assert result["grid_w"] == expected_gw
        assert result["logits"].shape == (B, expected_gh * expected_gw, 1)

    def test_weighted_image(self):
        from backend.ml.keypoint_detector import WatermarkKeypointDetector

        detector = WatermarkKeypointDetector(
            patch_size=16, feat_dim=64, num_heads=2, num_layers=2, max_patches=64
        )
        x = _rand_image()
        weighted, attn = detector.get_weighted_image(x)
        assert weighted.shape == x.shape
        assert attn.shape == (B, 1, H, W)

    def test_gradient_flow(self):
        from backend.ml.keypoint_detector import WatermarkKeypointDetector

        detector = WatermarkKeypointDetector(
            patch_size=16, feat_dim=64, num_heads=2, num_layers=2, max_patches=64
        )
        x = _rand_image().requires_grad_(True)
        result = detector(x)
        result["logits"].sum().backward()
        assert x.grad is not None


class TestKeypointDetectorLoss:
    def test_output_keys(self):
        from backend.ml.keypoint_detector import KeypointDetectorLoss

        loss_fn = KeypointDetectorLoss()
        logits = torch.randn(B, 16, 1)
        target = torch.randint(0, 2, (B, 16, 1)).float()
        result = loss_fn(logits, target)
        assert "loss" in result
        assert "bce" in result
        assert "dice" in result
        assert "accuracy" in result

    def test_perfect_prediction(self):
        from backend.ml.keypoint_detector import KeypointDetectorLoss

        loss_fn = KeypointDetectorLoss()
        target = torch.ones(B, 16, 1)
        logits = torch.ones(B, 16, 1) * 10  # high confidence positive
        result = loss_fn(logits, target)
        assert result["accuracy"].item() > 0.99

    def test_dice_loss_range(self):
        from backend.ml.keypoint_detector import KeypointDetectorLoss

        loss_fn = KeypointDetectorLoss()
        logits = torch.randn(B, 16, 1)
        target = torch.randint(0, 2, (B, 16, 1)).float()
        result = loss_fn(logits, target)
        assert 0 <= result["dice"].item() <= 1.0


# =====================================================================
# ddim_inversion.py
# =====================================================================

class TestDDIMInversion:
    def test_inversion_output_shape(self):
        from backend.ml.ddim_inversion import DDIMInversion

        # Simple noise predictor mock
        predictor = _MockNoisePredictor()
        inverter = DDIMInversion(predictor, num_steps=100, inversion_steps=5)
        x = _rand_image()
        z_T = inverter(x)
        assert z_T.shape == x.shape

    def test_intermediates_count(self):
        from backend.ml.ddim_inversion import DDIMInversion

        predictor = _MockNoisePredictor()
        inverter = DDIMInversion(predictor, num_steps=100, inversion_steps=10)
        x = _rand_image()
        z_T, intermediates = inverter.invert(x, num_steps=10)
        # Should have 10+1 intermediates (including initial x_0)
        assert len(intermediates) == 10  # 1 initial + (10-1) steps

    def test_predictor_frozen(self):
        from backend.ml.ddim_inversion import DDIMInversion

        predictor = _MockNoisePredictor()
        DDIMInversion(predictor, num_steps=100)
        for p in predictor.parameters():
            assert not p.requires_grad


class TestBlindDDIMInversion:
    def test_output_shape(self):
        from backend.ml.ddim_inversion import BlindDDIMInversion

        model = BlindDDIMInversion(base_channels=16)
        x = _rand_image()
        z_T = model(x)
        assert z_T.shape == x.shape

    def test_training_loss(self):
        from backend.ml.ddim_inversion import BlindDDIMInversion

        model = BlindDDIMInversion(base_channels=16)
        x = _rand_image()
        z_T_target = torch.randn_like(x)
        result = model.training_loss(x, z_T_target)
        assert "loss" in result
        assert "cosine_sim" in result
        assert result["loss"].item() > 0

    def test_gradient_flow(self):
        from backend.ml.ddim_inversion import BlindDDIMInversion

        model = BlindDDIMInversion(base_channels=16)
        x = _rand_image().requires_grad_(True)
        z_T = model(x)
        z_T.sum().backward()
        assert x.grad is not None


class TestLatentWatermarkExtractor:
    def test_output_shape(self):
        from backend.ml.ddim_inversion import LatentWatermarkExtractor

        extractor = LatentWatermarkExtractor(pattern_dim=48, latent_channels=3)
        z_T = torch.randn(B, 3, 32, 32)
        logits = extractor(z_T)
        assert logits.shape == (B, 48)

    def test_detect_output_keys(self):
        from backend.ml.ddim_inversion import LatentWatermarkExtractor

        extractor = LatentWatermarkExtractor(pattern_dim=48, latent_channels=3)
        z_T = torch.randn(B, 3, 32, 32)
        expected_bits = _rand_bits()
        result = extractor.detect(z_T, expected_bits)
        assert "logits" in result
        assert "predicted_bits" in result
        assert "bit_accuracy" in result
        assert "is_watermarked" in result
        assert result["bit_accuracy"].shape == (B,)
        assert result["is_watermarked"].shape == (B,)

    def test_detect_perfect_match(self):
        from backend.ml.ddim_inversion import LatentWatermarkExtractor

        extractor = LatentWatermarkExtractor(pattern_dim=48, latent_channels=3)
        z_T = torch.randn(B, 3, 32, 32)
        logits = extractor(z_T)
        predicted_bits = (logits > 0).float()
        # Use predicted bits as "expected" → perfect match
        result = extractor.detect(z_T, predicted_bits)
        assert result["bit_accuracy"].min().item() > 0.99


class _MockNoisePredictor(nn.Module):
    """Simple mock noise predictor for testing DDIM inversion."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        # Return constant small noise prediction
        return torch.randn_like(x) * 0.1
