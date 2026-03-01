"""Lazy-loading singleton for ML models.

Models are loaded once on first use and kept in memory for the lifetime of
the Celery worker process.  Weights are loaded from ``settings.upload_dir /
"weights"`` — if no checkpoint exists the models are initialised with random
weights (useful during development before training).
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock

import torch

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_lock = Lock()
_encoder: torch.nn.Module | None = None
_decoder: torch.nn.Module | None = None
_igrm: torch.nn.Module | None = None

WEIGHTS_DIR = Path(settings.upload_dir) / "weights"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_checkpoint(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """Try to load ``<WEIGHTS_DIR>/<name>.pt``; fall back to random init."""
    ckpt = WEIGHTS_DIR / f"{name}.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        logger.info("Loaded weights from %s", ckpt)
    else:
        logger.warning("No checkpoint at %s — using random weights", ckpt)
    return model.to(DEVICE).eval()


def get_encoder() -> torch.nn.Module:
    global _encoder
    if _encoder is None:
        with _lock:
            if _encoder is None:
                from backend.ml.encoder import HiDDeNEncoder

                _encoder = _load_checkpoint(HiDDeNEncoder(message_length=48), "encoder")
    return _encoder


def get_decoder() -> torch.nn.Module:
    global _decoder
    if _decoder is None:
        with _lock:
            if _decoder is None:
                from backend.ml.decoder import SwinWatermarkDecoder

                _decoder = _load_checkpoint(SwinWatermarkDecoder(message_length=48), "decoder")
    return _decoder


def get_igrm() -> torch.nn.Module:
    global _igrm
    if _igrm is None:
        with _lock:
            if _igrm is None:
                from backend.ml.igrm import IGRM

                _igrm = _load_checkpoint(IGRM(), "igrm")
    return _igrm
