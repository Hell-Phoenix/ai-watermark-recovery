"""Celery tasks for image processing (watermark embed / extract / recover).

Each task:
1. Marks the DB job as RUNNING.
2. Loads the image → tensor.
3. Runs the appropriate ML model(s).
4. Persists results (image and/or JSON metadata).
5. Marks the DB job as SUCCESS (or FAILURE via _BaseTask.on_failure).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from celery import Task
from PIL import Image as PILImage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from torchvision import transforms
from torchvision.utils import save_image

from backend.app.core.config import get_settings
from backend.app.models.job import Job, JobStatus
from backend.app.worker import celery_app

logger = logging.getLogger(__name__)
settings = get_settings()

# --------------- constants ---------------

MESSAGE_LENGTH = 48  # bits — must match encoder / decoder config
_IMG_SIZE = 256  # H/W the encoder expects (square)

# directories
_RESULTS_DIR = Path(settings.upload_dir) / "results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Celery tasks run in a sync worker — use a sync engine.
_sync_engine = create_engine(settings.database_url_sync, pool_pre_ping=True)


def _get_sync_session() -> Session:
    return Session(bind=_sync_engine)


# --------------- tensor helpers ---------------

_to_tensor = transforms.Compose([
    transforms.Resize((_IMG_SIZE, _IMG_SIZE)),
    transforms.ToTensor(),  # → [0, 1] float32
])


def _load_image_tensor(path: str) -> torch.Tensor:
    """Load an image file and return a (1, 3, H, W) tensor on the model device."""
    from backend.ml.model_loader import DEVICE

    img = PILImage.open(path).convert("RGB")
    return _to_tensor(img).unsqueeze(0).to(DEVICE)


def _hex_to_bits(hex_str: str) -> torch.Tensor:
    """Convert a hex string to a (1, MESSAGE_LENGTH) float tensor of 0/1."""
    from backend.ml.model_loader import DEVICE

    # Pad or truncate to the exact number of hex chars needed
    n_hex = MESSAGE_LENGTH // 4  # 48 bits → 12 hex chars
    hex_str = hex_str.ljust(n_hex, "0")[:n_hex]
    bits_int = bin(int(hex_str, 16))[2:].zfill(MESSAGE_LENGTH)
    bits = [float(b) for b in bits_int]
    return torch.tensor(bits, dtype=torch.float32, device=DEVICE).unsqueeze(0)


def _bits_to_hex(bits: torch.Tensor) -> str:
    """Convert a (MESSAGE_LENGTH,) binary tensor to a hex string."""
    bit_str = "".join(str(int(b)) for b in bits)
    return hex(int(bit_str, 2))[2:].zfill(MESSAGE_LENGTH // 4)


# --------------- base task ---------------

class _BaseTask(Task):
    """Abstract base task with automatic DB status bookkeeping."""

    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):  # noqa: ANN001
        job_id: str | None = kwargs.get("job_id") or (args[0] if args else None)
        if job_id:
            with _get_sync_session() as session:
                job = session.get(Job, uuid.UUID(job_id))
                if job:
                    job.status = JobStatus.FAILURE
                    job.error_message = str(exc)
                    job.finished_at = datetime.now(timezone.utc)
                    session.commit()
        logger.exception("Task %s failed", task_id)


# --------------- tasks ---------------


@celery_app.task(bind=True, base=_BaseTask, name="tasks.embed_watermark")
def embed_watermark(
    self: Task,
    job_id: str,
    image_path: str,
    payload_hex: str = "000000000000",
) -> dict:
    """Embed a watermark into the given image.

    Parameters
    ----------
    job_id : str
        UUID of the ``Job`` row.
    image_path : str
        Path to the cover image.
    payload_hex : str
        Hex-encoded payload (up to 12 hex chars = 48 bits).
    """
    _update_status(job_id, JobStatus.RUNNING)

    from backend.ml.model_loader import get_encoder

    encoder = get_encoder()
    cover = _load_image_tensor(image_path)
    message = _hex_to_bits(payload_hex)

    with torch.no_grad():
        watermarked = encoder(cover, message)  # (1, 3, H, W)

    # Save the watermarked image
    out_name = f"embedded_{uuid.uuid4().hex[:8]}_{Path(image_path).stem}.png"
    result_path = _RESULTS_DIR / out_name
    save_image(watermarked.squeeze(0).cpu(), str(result_path))

    _update_status(job_id, JobStatus.SUCCESS, result_path=str(result_path))
    logger.info("embed_watermark complete — %s → %s", image_path, result_path)
    return {"job_id": job_id, "result_path": str(result_path)}


@celery_app.task(bind=True, base=_BaseTask, name="tasks.extract_watermark")
def extract_watermark(self: Task, job_id: str, image_path: str) -> dict:
    """Extract (detect) a watermark payload from the given image.

    Returns a dict with ``payload_hex`` and per-bit ``confidence``.
    """
    _update_status(job_id, JobStatus.RUNNING)

    from backend.ml.model_loader import get_decoder

    decoder = get_decoder()
    image_t = _load_image_tensor(image_path)

    with torch.no_grad():
        logits = decoder(image_t)  # (1, MESSAGE_LENGTH)

    probs = torch.sigmoid(logits).squeeze(0)  # (MESSAGE_LENGTH,)
    predicted_bits = (probs > 0.5).float()
    confidence = float(probs.mean())  # average bit probability

    payload_hex = _bits_to_hex(predicted_bits)

    # Persist metadata as JSON sidecar
    meta = {
        "payload_hex": payload_hex,
        "confidence": round(confidence, 4),
        "bits": [int(b) for b in predicted_bits.tolist()],
    }
    meta_path = _RESULTS_DIR / f"detect_{uuid.uuid4().hex[:8]}.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    _update_status(job_id, JobStatus.SUCCESS, result_path=str(meta_path))
    logger.info("extract_watermark complete — confidence=%.4f", confidence)
    return {"job_id": job_id, "payload_hex": payload_hex, "confidence": confidence}


@celery_app.task(bind=True, base=_BaseTask, name="tasks.recover_watermark")
def recover_watermark(self: Task, job_id: str, image_path: str) -> dict:
    """Run IGRM-based restoration on a degraded image, then decode the watermark.

    Pipeline: degraded → IGRM (restore) → decoder (extract payload).
    """
    _update_status(job_id, JobStatus.RUNNING)

    from backend.ml.model_loader import get_decoder, get_igrm

    igrm = get_igrm()
    decoder = get_decoder()
    degraded = _load_image_tensor(image_path)

    with torch.no_grad():
        restored = igrm(degraded)       # (1, 3, H, W)
        logits = decoder(restored)       # (1, MESSAGE_LENGTH)

    probs = torch.sigmoid(logits).squeeze(0)
    predicted_bits = (probs > 0.5).float()
    confidence = float(probs.mean())
    payload_hex = _bits_to_hex(predicted_bits)

    # Save the restored image
    restored_name = f"restored_{uuid.uuid4().hex[:8]}_{Path(image_path).stem}.png"
    restored_path = _RESULTS_DIR / restored_name
    save_image(restored.squeeze(0).cpu(), str(restored_path))

    # Persist metadata
    meta = {
        "payload_hex": payload_hex,
        "confidence": round(confidence, 4),
        "bits": [int(b) for b in predicted_bits.tolist()],
        "restored_image": str(restored_path),
    }
    meta_path = _RESULTS_DIR / f"recover_{uuid.uuid4().hex[:8]}.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    _update_status(job_id, JobStatus.SUCCESS, result_path=str(meta_path))
    logger.info("recover_watermark complete — confidence=%.4f", confidence)
    return {
        "job_id": job_id,
        "payload_hex": payload_hex,
        "confidence": confidence,
        "restored_image": str(restored_path),
    }


@celery_app.task(bind=True, base=_BaseTask, name="tasks.detect_forgery")
def detect_forgery(self: Task, job_id: str, image_path: str) -> dict:
    """Detect whether an image has been forged or tampered with.

    Pipeline: image → IGRM (restore) → compare original vs restored.
    A large reconstruction residual indicates the image has been attacked;
    a small residual means it is likely clean or only lightly compressed.

    Returns a dict with ``forgery_detected`` (bool) and ``confidence`` (float).
    """
    _update_status(job_id, JobStatus.RUNNING)

    from backend.ml.model_loader import get_decoder, get_igrm

    igrm = get_igrm()
    decoder = get_decoder()
    image_t = _load_image_tensor(image_path)

    with torch.no_grad():
        restored = igrm(image_t)  # (1, 3, H, W)

        # Measure reconstruction residual — high residual ≈ heavy attack
        residual = (restored - image_t).abs()
        residual_score = float(residual.mean())

        # Extract watermark from both original and restored
        logits_orig = decoder(image_t)    # (1, MESSAGE_LENGTH)
        logits_rest = decoder(restored)   # (1, MESSAGE_LENGTH)

        probs_orig = torch.sigmoid(logits_orig).squeeze(0)
        probs_rest = torch.sigmoid(logits_rest).squeeze(0)

        bits_orig = (probs_orig > 0.5).float()
        bits_rest = (probs_rest > 0.5).float()

        # Bit agreement between original and restored extractions
        bit_agreement = float((bits_orig == bits_rest).float().mean())

        # Confidence that forgery occurred:
        #   high residual + low bit agreement → likely forged
        #   low residual + high bit agreement → likely clean
        # Normalise residual_score to [0, 1] with a sigmoid-like mapping
        residual_confidence = 1.0 - (1.0 / (1.0 + 10.0 * residual_score))
        disagreement_confidence = 1.0 - bit_agreement

        # Weighted combination
        confidence = 0.6 * residual_confidence + 0.4 * disagreement_confidence
        forgery_detected = confidence > 0.5

    # Persist metadata
    meta = {
        "forgery_detected": forgery_detected,
        "confidence": round(confidence, 4),
        "residual_score": round(residual_score, 6),
        "bit_agreement": round(bit_agreement, 4),
        "payload_hex_original": _bits_to_hex(bits_orig),
        "payload_hex_restored": _bits_to_hex(bits_rest),
    }
    meta_path = _RESULTS_DIR / f"forgery_{uuid.uuid4().hex[:8]}.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    _update_status(job_id, JobStatus.SUCCESS, result_path=str(meta_path))
    logger.info(
        "detect_forgery complete — forgery=%s confidence=%.4f",
        forgery_detected, confidence,
    )
    return {
        "job_id": job_id,
        "forgery_detected": forgery_detected,
        "confidence": confidence,
    }


# --------------- helpers ---------------


def _update_status(
    job_id: str,
    status: JobStatus,
    *,
    result_path: str | None = None,
) -> None:
    with _get_sync_session() as session:
        job = session.get(Job, uuid.UUID(job_id))
        if job:
            job.status = status
            if result_path:
                job.result_path = result_path
            if status in {JobStatus.SUCCESS, JobStatus.FAILURE}:
                job.finished_at = datetime.now(timezone.utc)
            session.commit()
