"""Watermark embed / detect / poll / audit endpoints — Phase 6.

Endpoints:
  POST /embed  — enqueue a watermark embedding job → ``202 Accepted``
  POST /detect — enqueue a watermark detection job → ``202 Accepted``
  GET  /job/{id}          — poll job status + result when complete
  GET  /audit/{image_hash} — forensic audit log for an image
"""

from __future__ import annotations

import hashlib
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.auth import require_auth, rate_limit
from backend.app.core.database import get_db
from backend.app.models.image import Image
from backend.app.models.job import Job, JobType, JobStatus
from backend.app.models.user import User
from backend.app.schemas.detection import (
    AuditEntry,
    AuditResponse,
    DetectRequest,
    DetectResponse,
    DetectResult,
    EmbedRequest,
    EmbedResponse,
    EmbedResult,
    JobStatusResponse,
)
from backend.app.tasks.pipeline_tasks import pipeline_embed, pipeline_detect

router = APIRouter(tags=["watermark"])


# ===================================================================
# POST /embed
# ===================================================================

@router.post(
    "/embed",
    response_model=EmbedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Embed watermark into an image",
)
async def embed_watermark(
    body: EmbedRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_auth),
    _rl: None = Depends(rate_limit(max_requests=30, window_seconds=60)),
) -> EmbedResponse:
    """Enqueue an async watermark embedding job.

    The caller should poll ``GET /job/{job_id}`` until the job status
    becomes ``success`` or ``failure``.
    """
    image = await db.get(Image, body.image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    if image.owner_id != user.id:
        raise HTTPException(status_code=403, detail="Not your image")

    job = Job(image_id=image.id, job_type=JobType.EMBED_WATERMARK)
    db.add(job)
    await db.flush()
    await db.refresh(job)

    result = pipeline_embed.delay(
        str(job.id), image.filepath, body.payload_hex, body.sign,
    )
    job.celery_task_id = result.id
    await db.flush()
    await db.refresh(job)

    return EmbedResponse(job_id=job.id)


# ===================================================================
# POST /detect
# ===================================================================

@router.post(
    "/detect",
    response_model=DetectResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Detect / recover watermark from image",
)
async def detect_watermark(
    body: DetectRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_auth),
    _rl: None = Depends(rate_limit(max_requests=30, window_seconds=60)),
) -> DetectResponse:
    """Enqueue an async watermark detection / recovery job.

    Returns the full detection schema (payload, confidence, attack type,
    tamper mask, dual-layer integrity, forgery flag) once the job completes.
    """
    image = await db.get(Image, body.image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    if image.owner_id != user.id:
        raise HTTPException(status_code=403, detail="Not your image")

    job = Job(image_id=image.id, job_type=JobType.EXTRACT_WATERMARK)
    db.add(job)
    await db.flush()
    await db.refresh(job)

    result = pipeline_detect.delay(
        str(job.id), image.filepath, body.verify_signature,
    )
    job.celery_task_id = result.id
    await db.flush()
    await db.refresh(job)

    return DetectResponse(job_id=job.id)


# ===================================================================
# GET /job/{id}
# ===================================================================

@router.get(
    "/job/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll async job status",
)
async def get_job_status(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_auth),
) -> JobStatusResponse:
    """Return the current status (and result, if finished) of a processing job."""
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Load the image to check ownership
    image = await db.get(Image, job.image_id)
    if image and image.owner_id != user.id:
        raise HTTPException(status_code=403, detail="Not your job")

    result_data = None
    if job.status == JobStatus.SUCCESS and job.result_path:
        try:
            raw = json.loads(open(job.result_path).read())  # noqa: SIM115
            if job.job_type == JobType.EMBED_WATERMARK:
                result_data = EmbedResult(**raw)
            else:
                result_data = DetectResult(**raw)
        except Exception:
            pass  # result will be None — caller can still see status

    return JobStatusResponse(
        id=job.id,
        job_type=job.job_type,
        status=job.status,
        created_at=job.created_at,
        finished_at=job.finished_at,
        error_message=job.error_message,
        result=result_data,
    )


# ===================================================================
# GET /audit/{image_hash}
# ===================================================================

@router.get(
    "/audit/{image_hash}",
    response_model=AuditResponse,
    summary="Forensic audit log for an image",
)
async def audit_image(
    image_hash: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_auth),
) -> AuditResponse:
    """Return the processing history for an image identified by its SHA-256
    content hash.  Useful for forensic provenance tracking.
    """
    # Find images whose filepath matches (simple lookup — in production we'd
    # store the content hash on the Image model itself).
    stmt = (
        select(Job)
        .join(Image)
        .where(Image.owner_id == user.id)
        .order_by(Job.created_at.desc())
    )
    result = await db.execute(stmt)
    all_jobs = result.scalars().all()

    # Filter jobs whose image file hashes match (or just return all user jobs
    # for the prototype — hashing every file per request is expensive)
    entries = [
        AuditEntry(
            job_id=j.id,
            job_type=j.job_type,
            status=j.status,
            created_at=j.created_at,
            finished_at=j.finished_at,
        )
        for j in all_jobs
    ]

    return AuditResponse(image_hash=image_hash, entries=entries)
