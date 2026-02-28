"""Job schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel

from backend.app.models.job import JobType


class JobCreate(BaseModel):
    image_id: uuid.UUID
    job_type: JobType


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
