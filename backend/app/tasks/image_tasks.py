"""Celery tasks for image processing (watermark embed / extract / recover)."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from celery import Task
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.models.job import Job, JobStatus
from backend.app.worker import celery_app

logger = logging.getLogger(__name__)
settings = get_settings()

# Celery tasks run in a sync worker — use a sync engine.
_sync_engine = create_engine(settings.database_url_sync, pool_pre_ping=True)


def _get_sync_session() -> Session:
    return Session(bind=_sync_engine)


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


@celery_app.task(bind=True, base=_BaseTask, name="tasks.embed_watermark")
def embed_watermark(self: Task, job_id: str, image_path: str) -> dict:
    """Embed a watermark into the given image."""
    _update_status(job_id, JobStatus.RUNNING)

    # ---- placeholder: call ML pipeline here ----
    output_dir = Path(settings.upload_dir) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"embedded_{Path(image_path).name}"

    # Simulate processing (replace with real ML encoder call)
    from PIL import Image as PILImage

    img = PILImage.open(image_path)
    img.save(str(result_path))
    # ---- end placeholder ----

    _update_status(job_id, JobStatus.SUCCESS, result_path=str(result_path))
    return {"job_id": job_id, "result_path": str(result_path)}


@celery_app.task(bind=True, base=_BaseTask, name="tasks.extract_watermark")
def extract_watermark(self: Task, job_id: str, image_path: str) -> dict:
    """Extract a watermark from the given image."""
    _update_status(job_id, JobStatus.RUNNING)

    # Placeholder — replace with ml.decoder call
    result = {"job_id": job_id, "watermark_bits": "0" * 64, "confidence": 0.0}

    _update_status(job_id, JobStatus.SUCCESS)
    return result


@celery_app.task(bind=True, base=_BaseTask, name="tasks.recover_watermark")
def recover_watermark(self: Task, job_id: str, image_path: str) -> dict:
    """Run IGRM-based watermark recovery on a degraded image."""
    _update_status(job_id, JobStatus.RUNNING)

    # Placeholder — replace with ml.igrm call
    result = {"job_id": job_id, "recovered": True}

    _update_status(job_id, JobStatus.SUCCESS)
    return result


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
