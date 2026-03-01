"""Job endpoints — create processing jobs & poll status."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.models.image import Image
from backend.app.models.job import Job, JobType
from backend.app.schemas.job import JobCreate, JobOut
from backend.app.tasks.image_tasks import (
    detect_forgery,
    embed_watermark,
    extract_watermark,
    recover_watermark,
)

router = APIRouter(prefix="/jobs", tags=["jobs"])

_TASK_MAP = {
    JobType.EMBED_WATERMARK: embed_watermark,
    JobType.EXTRACT_WATERMARK: extract_watermark,
    JobType.RECOVER_WATERMARK: recover_watermark,
    JobType.DETECT_FORGERY: detect_forgery,
}


@router.post("/", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    payload: JobCreate,
    db: AsyncSession = Depends(get_db),
) -> JobOut:
    """Submit an async image-processing job backed by Celery."""
    image = await db.get(Image, payload.image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    task_fn = _TASK_MAP.get(payload.job_type)
    if not task_fn:
        raise HTTPException(status_code=400, detail=f"Unsupported job type: {payload.job_type}")

    job = Job(image_id=image.id, job_type=payload.job_type)
    db.add(job)
    await db.flush()
    await db.refresh(job)

    # Dispatch to Celery — embed tasks also receive the payload
    if payload.job_type == JobType.EMBED_WATERMARK:
        result = task_fn.delay(str(job.id), image.filepath, payload.payload_hex or "000000000000")
    else:
        result = task_fn.delay(str(job.id), image.filepath)
    job.celery_task_id = result.id
    await db.flush()
    await db.refresh(job)

    return JobOut.model_validate(job)


@router.get("/{job_id}", response_model=JobOut)
async def get_job(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> JobOut:
    """Poll job status."""
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobOut.model_validate(job)


@router.get("/", response_model=list[JobOut])
async def list_jobs(
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
) -> list[JobOut]:
    result = await db.execute(select(Job).offset(skip).limit(limit))
    return [JobOut.model_validate(j) for j in result.scalars().all()]
