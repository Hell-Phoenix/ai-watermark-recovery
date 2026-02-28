"""Job schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel

from backend.app.models.job import JobType


class JobCreate(BaseModel):
    image_id: uuid.UUID
    job_type: JobType
    payload_hex: str | None = None  # required for embed_watermark (up to 12 hex chars = 48 bits)


class JobOut(BaseModel):
    id: uuid.UUID
    celery_task_id: str | None
    job_type: str
    status: str
    result_path: str | None
    error_message: str | None
    created_at: datetime
    finished_at: datetime | None

    model_config = {"from_attributes": True}
