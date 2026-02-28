"""Curriculum learning scheduler for the Attack Simulation Layer.

Controls the ASL ``severity`` parameter over the course of training so the
watermark network starts with easy attacks and gradually advances to hard
ones.  This lets the encoder-decoder find a stable gradient signal early
before being pushed toward extreme robustness.

Schedules
---------
``LinearCurriculum``
    Severity increases linearly from 0 to 1 over a configurable fraction of
    total training.

``CosineCurriculum``
    Severity follows a half-cosine ramp (slow start, fast middle, slow end).

``StepCurriculum``
    Severity increases in discrete steps at defined epoch boundaries.

Usage
-----
>>> from backend.ml.asl import AttackSimulationLayer
>>> from backend.training.curriculum import LinearCurriculum
>>> asl = AttackSimulationLayer()
>>> scheduler = LinearCurriculum(asl, total_epochs=100, warmup_fraction=0.1, ramp_fraction=0.8)
>>> for epoch in range(100):
...     scheduler.step(epoch)
...     # asl.severity is now automatically updated
...     for batch in dataloader:
...         attacked = asl(watermarked_batch)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from backend.ml.asl import AttackSimulationLayer


class BaseCurriculum(ABC):
    """Abstract base class for curriculum schedulers.

    Parameters
    ----------
    asl : AttackSimulationLayer
        The attack layer whose ``severity`` will be controlled.
    total_epochs : int
        Total number of training epochs.
    """

    def __init__(self, asl: AttackSimulationLayer, total_epochs: int) -> None:
        self.asl = asl
        self.total_epochs = total_epochs
        self._current_epoch = 0

    @abstractmethod
    def _compute_severity(self, epoch: int) -> float:
        """Return the target severity for the given epoch."""
        ...

    def step(self, epoch: int | None = None) -> float:
        """Advance the curriculum and update the ASL severity.

        Parameters
        ----------
        epoch : int | None
            Current epoch (0-indexed).  If *None* the internal counter is
            incremented automatically.

        Returns
        -------
        float
            The new severity value.
        """
        if epoch is not None:
            self._current_epoch = epoch
        severity = self._compute_severity(self._current_epoch)
        self.asl.severity = severity
        if epoch is None:
            self._current_epoch += 1
        return severity

    @property
    def severity(self) -> float:
        return self.asl.severity

    def state_dict(self) -> dict:
        return {"current_epoch": self._current_epoch}

    def load_state_dict(self, state: dict) -> None:
        self._current_epoch = state["current_epoch"]


# ────────────────────────────────────────────────────────────


class LinearCurriculum(BaseCurriculum):
    """Severity ramps linearly from 0 to 1.

    ::

        severity
        1.0 ┤                         ┌───────────
            │                        ╱
            │                       ╱
        0.0 ┤───────────────────────╱
            └──────────────────────────────────── epoch
               warmup          ramp        hold

    Parameters
    ----------
    warmup_fraction : float
        Fraction of total_epochs where severity stays at 0 (default 0.1).
    ramp_fraction : float
        Fraction of total_epochs for the linear ramp (default 0.7).
        After ``warmup + ramp`` epochs, severity is held at 1.0.
    """

    def __init__(
        self,
        asl: AttackSimulationLayer,
        total_epochs: int,
        warmup_fraction: float = 0.1,
        ramp_fraction: float = 0.7,
    ) -> None:
        super().__init__(asl, total_epochs)
        self.warmup_fraction = warmup_fraction
        self.ramp_fraction = ramp_fraction

    def _compute_severity(self, epoch: int) -> float:
        warmup_end = int(self.total_epochs * self.warmup_fraction)
        ramp_end = warmup_end + int(self.total_epochs * self.ramp_fraction)

        if epoch < warmup_end:
            return 0.0
        if epoch >= ramp_end:
            return 1.0
        # Linear interpolation within the ramp
        progress = (epoch - warmup_end) / max(ramp_end - warmup_end, 1)
        return float(progress)


class CosineCurriculum(BaseCurriculum):
    """Severity follows a half-cosine curve (slow → fast → slow).

    ::

        severity
        1.0 ┤                              ╭─────
            │                           ╭──╯
            │                      ╭───╯
            │               ╭─────╯
        0.0 ┤──────────────╯
            └──────────────────────────────────── epoch
              warmup             cosine ramp

    Parameters
    ----------
    warmup_fraction : float
        Fraction of total_epochs where severity stays at 0 (default 0.1).
    """

    def __init__(
        self,
        asl: AttackSimulationLayer,
        total_epochs: int,
        warmup_fraction: float = 0.1,
    ) -> None:
        super().__init__(asl, total_epochs)
        self.warmup_fraction = warmup_fraction

    def _compute_severity(self, epoch: int) -> float:
        warmup_end = int(self.total_epochs * self.warmup_fraction)
        if epoch < warmup_end:
            return 0.0
        remaining = self.total_epochs - warmup_end
        progress = min((epoch - warmup_end) / max(remaining, 1), 1.0)
        # Half-cosine: 0 → 1
        return float(0.5 * (1.0 - math.cos(math.pi * progress)))


class StepCurriculum(BaseCurriculum):
    """Severity increases in discrete steps at specified epoch thresholds.

    Parameters
    ----------
    milestones : list[tuple[int, float]]
        Sorted list of ``(epoch, severity)`` pairs.  Between milestones
        the severity is held constant.

    Example
    -------
    >>> scheduler = StepCurriculum(asl, total_epochs=100, milestones=[
    ...     (0, 0.0), (20, 0.2), (40, 0.5), (60, 0.8), (80, 1.0)
    ... ])
    """

    def __init__(
        self,
        asl: AttackSimulationLayer,
        total_epochs: int,
        milestones: list[tuple[int, float]] | None = None,
    ) -> None:
        super().__init__(asl, total_epochs)
        if milestones is None:
            # Default: 5 equal steps
            n_steps = 5
            step_size = total_epochs // n_steps
            milestones = [(i * step_size, i / (n_steps - 1)) for i in range(n_steps)]
        self.milestones = sorted(milestones, key=lambda m: m[0])

    def _compute_severity(self, epoch: int) -> float:
        severity = 0.0
        for ep, sev in self.milestones:
            if epoch >= ep:
                severity = sev
            else:
                break
        return float(severity)
