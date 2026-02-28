"""Tests for Phase 2 — Attack Simulation Layer modules.

Covers:
  - DifferentiableJPEG  (jsnet.py)
  - DifferentiableCrop / GeometricDistortion  (stn_crop.py)
  - GaussianNoise / GaussianBlur / DiffusionNoiseAttack  (asl.py components)
  - AttackSimulationLayer  (asl.py orchestrator)
  - Curriculum schedulers  (training/curriculum.py)
"""

from __future__ import annotations

import pytest
import torch

# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────


def _rand_image(batch: int = 2, h: int = 64, w: int = 64) -> torch.Tensor:
    """Random RGB tensor in [0, 1]."""
    return torch.rand(batch, 3, h, w)


# ────────────────────────────────────────────────────────────
# DifferentiableJPEG
# ────────────────────────────────────────────────────────────


class TestDifferentiableJPEG:
    def test_output_shape(self) -> None:
        from backend.ml.jsnet import DifferentiableJPEG

        jpeg = DifferentiableJPEG(quality=50)
        x = _rand_image(2, 64, 64)
        out = jpeg(x)
        assert out.shape == x.shape

    def test_output_range(self) -> None:
        from backend.ml.jsnet import DifferentiableJPEG

        jpeg = DifferentiableJPEG(quality=30)
        out = jpeg(_rand_image())
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_quality_override(self) -> None:
        """Higher QF should produce less distortion than lower QF."""
        from backend.ml.jsnet import DifferentiableJPEG

        jpeg = DifferentiableJPEG(quality=50, use_ste=True)
        # Use a structured (non-random-noise) image so low-frequency
        # content survives high-QF quantization, giving a clear MSE gap.
        torch.manual_seed(42)
        base = torch.rand(1, 3, 1, 1).expand(1, 3, 64, 64).clone()
        for s in [16, 8, 4]:
            patch = torch.randn(1, 3, 64 // s, 64 // s)
            patch = torch.nn.functional.interpolate(patch, size=(64, 64), mode="bilinear")
            base = base + patch * (s / 64.0)
        x = base.clamp(0.0, 1.0)

        out_high = jpeg(x, quality=98)
        out_low = jpeg(x, quality=80)
        mse_high = (out_high - x).pow(2).mean()
        mse_low = (out_low - x).pow(2).mean()
        assert mse_low > mse_high, (
            f"Expected QF=80 MSE ({mse_low:.4f}) > QF=98 MSE ({mse_high:.4f})"
        )

    def test_gradient_flows(self) -> None:
        from backend.ml.jsnet import DifferentiableJPEG

        jpeg = DifferentiableJPEG(quality=50, use_ste=False)
        x = _rand_image(1, 32, 32).requires_grad_(True)
        out = jpeg(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_non_multiple_of_8(self) -> None:
        """Images not divisible by 8 should still work (padding)."""
        from backend.ml.jsnet import DifferentiableJPEG

        jpeg = DifferentiableJPEG(quality=50)
        x = _rand_image(1, 30, 45)  # not multiples of 8
        out = jpeg(x)
        assert out.shape == x.shape

    def test_ste_mode(self) -> None:
        from backend.ml.jsnet import DifferentiableJPEG

        jpeg = DifferentiableJPEG(quality=50, use_ste=True)
        x = _rand_image(1, 32, 32).requires_grad_(True)
        out = jpeg(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ────────────────────────────────────────────────────────────
# DifferentiableCrop
# ────────────────────────────────────────────────────────────


class TestDifferentiableCrop:
    def test_output_shape_same(self) -> None:
        from backend.ml.stn_crop import DifferentiableCrop

        crop = DifferentiableCrop(min_ratio=0.3, max_ratio=0.8)
        crop.train()
        x = _rand_image(2, 64, 64)
        out = crop(x)
        assert out.shape == x.shape  # same H, W when output_size=None

    def test_output_shape_resized(self) -> None:
        from backend.ml.stn_crop import DifferentiableCrop

        crop = DifferentiableCrop(min_ratio=0.2, max_ratio=0.5, output_size=(32, 32))
        crop.train()
        x = _rand_image(2, 64, 64)
        out = crop(x)
        assert out.shape == (2, 3, 32, 32)

    def test_gradient_flows(self) -> None:
        from backend.ml.stn_crop import DifferentiableCrop

        crop = DifferentiableCrop(min_ratio=0.5, max_ratio=0.8)
        crop.train()
        x = _rand_image(1, 32, 32).requires_grad_(True)
        out = crop(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_full_crop_approximates_identity(self) -> None:
        from backend.ml.stn_crop import DifferentiableCrop

        crop = DifferentiableCrop(min_ratio=0.99, max_ratio=1.0)
        crop.train()
        x = _rand_image(1, 64, 64)
        out = crop(x)
        # Should be very close to input
        mse = (out - x).pow(2).mean()
        assert mse < 0.05


# ────────────────────────────────────────────────────────────
# GeometricDistortion
# ────────────────────────────────────────────────────────────


class TestGeometricDistortion:
    def test_output_shape(self) -> None:
        from backend.ml.stn_crop import GeometricDistortion

        geo = GeometricDistortion(max_angle=15, scale_range=(0.9, 1.1), max_shear=0.1)
        geo.train()
        x = _rand_image(2, 64, 64)
        out = geo(x)
        assert out.shape == x.shape

    def test_gradient_flows(self) -> None:
        from backend.ml.stn_crop import GeometricDistortion

        geo = GeometricDistortion()
        geo.train()
        x = _rand_image(1, 32, 32).requires_grad_(True)
        out = geo(x)
        out.sum().backward()
        assert x.grad is not None


# ────────────────────────────────────────────────────────────
# GaussianNoise
# ────────────────────────────────────────────────────────────


class TestGaussianNoise:
    def test_noise_added_in_train(self) -> None:
        from backend.ml.asl import GaussianNoise

        module = GaussianNoise(std=0.1)
        module.train()
        torch.manual_seed(42)
        x = _rand_image(1, 32, 32)
        out = module(x)
        assert not torch.allclose(out, x)

    def test_identity_in_eval(self) -> None:
        from backend.ml.asl import GaussianNoise

        module = GaussianNoise(std=0.1)
        module.eval()
        x = _rand_image(1, 32, 32)
        out = module(x)
        assert torch.allclose(out, x)

    def test_output_range(self) -> None:
        from backend.ml.asl import GaussianNoise

        module = GaussianNoise(std=0.5)
        module.train()
        out = module(_rand_image())
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ────────────────────────────────────────────────────────────
# GaussianBlur
# ────────────────────────────────────────────────────────────


class TestGaussianBlur:
    def test_blurs_image(self) -> None:
        from backend.ml.asl import GaussianBlur

        blur = GaussianBlur(kernel_size=7, sigma=2.0)
        x = _rand_image(1, 64, 64)
        out = blur(x)
        # Blurred image should be smoother → lower total variation
        def tv(img: torch.Tensor) -> float:
            dx = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()
            dy = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean()
            return float(dx + dy)

        assert tv(out) < tv(x)

    def test_gradient_flows(self) -> None:
        from backend.ml.asl import GaussianBlur

        blur = GaussianBlur()
        x = _rand_image(1, 32, 32).requires_grad_(True)
        out = blur(x)
        out.sum().backward()
        assert x.grad is not None


# ────────────────────────────────────────────────────────────
# DiffusionNoiseAttack
# ────────────────────────────────────────────────────────────


class TestDiffusionNoiseAttack:
    def test_output_shape(self) -> None:
        from backend.ml.asl import DiffusionNoiseAttack

        module = DiffusionNoiseAttack(max_timestep=100)
        module.train()
        x = _rand_image(2, 64, 64)
        out = module(x)
        assert out.shape == x.shape

    def test_noise_increases_mse(self) -> None:
        from backend.ml.asl import DiffusionNoiseAttack

        module = DiffusionNoiseAttack(max_timestep=500)
        module.train()
        x = _rand_image(4, 32, 32)
        out = module(x)
        mse = (out - x).pow(2).mean()
        assert mse > 0.0

    def test_identity_in_eval(self) -> None:
        from backend.ml.asl import DiffusionNoiseAttack

        module = DiffusionNoiseAttack(max_timestep=500)
        module.eval()
        x = _rand_image(1, 32, 32)
        out = module(x)
        assert torch.allclose(out, x)


# ────────────────────────────────────────────────────────────
# AttackSimulationLayer
# ────────────────────────────────────────────────────────────


class TestAttackSimulationLayer:
    def test_output_shape(self) -> None:
        from backend.ml.asl import AttackSimulationLayer

        asl = AttackSimulationLayer()
        asl.train()
        x = _rand_image(2, 64, 64)
        out = asl(x)
        assert out.shape == x.shape

    def test_identity_in_eval(self) -> None:
        from backend.ml.asl import AttackSimulationLayer

        asl = AttackSimulationLayer()
        asl.eval()
        x = _rand_image(2, 64, 64)
        out = asl(x)
        assert torch.allclose(out, x)

    def test_severity_property(self) -> None:
        from backend.ml.asl import AttackSimulationLayer

        asl = AttackSimulationLayer()
        asl.severity = 0.0
        assert asl.severity == 0.0
        asl.severity = 0.5
        assert abs(asl.severity - 0.5) < 1e-6
        asl.severity = 1.0
        assert asl.severity == 1.0

    def test_severity_clamps(self) -> None:
        from backend.ml.asl import AttackSimulationLayer

        asl = AttackSimulationLayer()
        asl.severity = -0.5
        assert asl.severity == 0.0
        asl.severity = 1.5
        assert asl.severity == 1.0

    def test_severity_updates_jpeg_qf(self) -> None:
        from backend.ml.asl import AttackSimulationLayer

        asl = AttackSimulationLayer()
        asl.severity = 0.0
        qf_easy = asl.jpeg.default_quality
        asl.severity = 1.0
        qf_hard = asl.jpeg.default_quality
        assert qf_easy > qf_hard

    def test_num_attacks(self) -> None:
        from backend.ml.asl import AttackSimulationLayer

        asl = AttackSimulationLayer(num_attacks=3)
        asl.train()
        x = _rand_image(2, 64, 64)
        # Should not crash with multiple sequential attacks
        out = asl(x)
        assert out.shape == x.shape

    def test_output_range(self) -> None:
        from backend.ml.asl import AttackSimulationLayer

        asl = AttackSimulationLayer(num_attacks=2)
        asl.train()
        asl.severity = 0.5
        x = _rand_image(4, 64, 64)
        out = asl(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ────────────────────────────────────────────────────────────
# Curriculum schedulers
# ────────────────────────────────────────────────────────────


class TestLinearCurriculum:
    def test_warmup_zero(self) -> None:
        from backend.ml.asl import AttackSimulationLayer
        from backend.training.curriculum import LinearCurriculum

        asl = AttackSimulationLayer()
        sched = LinearCurriculum(asl, total_epochs=100, warmup_fraction=0.1, ramp_fraction=0.7)
        sched.step(0)
        assert asl.severity == 0.0
        sched.step(5)
        assert asl.severity == 0.0  # still in warmup

    def test_ramp(self) -> None:
        from backend.ml.asl import AttackSimulationLayer
        from backend.training.curriculum import LinearCurriculum

        asl = AttackSimulationLayer()
        sched = LinearCurriculum(asl, total_epochs=100, warmup_fraction=0.1, ramp_fraction=0.7)
        # midpoint of ramp: epoch 45 → ~50% severity
        sched.step(45)
        assert 0.3 < asl.severity < 0.7

    def test_end_full(self) -> None:
        from backend.ml.asl import AttackSimulationLayer
        from backend.training.curriculum import LinearCurriculum

        asl = AttackSimulationLayer()
        sched = LinearCurriculum(asl, total_epochs=100, warmup_fraction=0.1, ramp_fraction=0.7)
        sched.step(99)
        assert asl.severity == 1.0

    def test_state_dict(self) -> None:
        from backend.ml.asl import AttackSimulationLayer
        from backend.training.curriculum import LinearCurriculum

        asl = AttackSimulationLayer()
        sched = LinearCurriculum(asl, total_epochs=100)
        sched.step(42)
        state = sched.state_dict()
        assert state["current_epoch"] == 42

        sched2 = LinearCurriculum(asl, total_epochs=100)
        sched2.load_state_dict(state)
        assert sched2._current_epoch == 42


class TestCosineCurriculum:
    def test_starts_at_zero(self) -> None:
        from backend.ml.asl import AttackSimulationLayer
        from backend.training.curriculum import CosineCurriculum

        asl = AttackSimulationLayer()
        sched = CosineCurriculum(asl, total_epochs=100, warmup_fraction=0.1)
        sched.step(0)
        assert asl.severity == 0.0

    def test_ends_at_one(self) -> None:
        from backend.ml.asl import AttackSimulationLayer
        from backend.training.curriculum import CosineCurriculum

        asl = AttackSimulationLayer()
        sched = CosineCurriculum(asl, total_epochs=100, warmup_fraction=0.0)
        sched.step(99)
        assert asl.severity > 0.99

    def test_monotonically_increasing(self) -> None:
        from backend.ml.asl import AttackSimulationLayer
        from backend.training.curriculum import CosineCurriculum

        asl = AttackSimulationLayer()
        sched = CosineCurriculum(asl, total_epochs=100, warmup_fraction=0.0)
        prev = -1.0
        for epoch in range(100):
            severity = sched.step(epoch)
            assert severity >= prev
            prev = severity


class TestStepCurriculum:
    def test_default_steps(self) -> None:
        from backend.ml.asl import AttackSimulationLayer
        from backend.training.curriculum import StepCurriculum

        asl = AttackSimulationLayer()
        sched = StepCurriculum(asl, total_epochs=100)
        sched.step(0)
        assert asl.severity == 0.0
        sched.step(99)
        assert asl.severity == 1.0

    def test_custom_milestones(self) -> None:
        from backend.ml.asl import AttackSimulationLayer
        from backend.training.curriculum import StepCurriculum

        asl = AttackSimulationLayer()
        milestones = [(0, 0.0), (30, 0.3), (60, 0.7), (80, 1.0)]
        sched = StepCurriculum(asl, total_epochs=100, milestones=milestones)
        sched.step(15)
        assert asl.severity == 0.0
        sched.step(35)
        assert abs(asl.severity - 0.3) < 1e-6
        sched.step(65)
        assert abs(asl.severity - 0.7) < 1e-6
        sched.step(90)
        assert asl.severity == 1.0
