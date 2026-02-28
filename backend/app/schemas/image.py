"""Image schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class ImageOut(BaseModel):
    id: uuid.UUID
    filename: str
    content_type: str | None
    file_size: int | None
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class ImageUploadResponse(BaseModel):
    image: ImageOut
    message: str = "Upload successful"
