"""Image upload & listing endpoints."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import get_settings
from backend.app.core.database import get_db
from backend.app.models.image import Image
from backend.app.schemas.image import ImageOut, ImageUploadResponse

router = APIRouter(prefix="/images", tags=["images"])
settings = get_settings()


@router.post("/", response_model=ImageUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_image(
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
    # owner_id would come from auth dependency in production
) -> ImageUploadResponse:
    """Upload an image file and persist metadata."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    dest = upload_dir / unique_name

    with dest.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)

    file_size = dest.stat().st_size

    # Placeholder owner — replace with real auth user
    placeholder_owner_id = uuid.UUID("00000000-0000-0000-0000-000000000000")

    image = Image(
        filename=file.filename or "unknown",
        filepath=str(dest),
        content_type=file.content_type,
        file_size=file_size,
        owner_id=placeholder_owner_id,
    )
    db.add(image)
    await db.flush()
    await db.refresh(image)

    return ImageUploadResponse(image=ImageOut.model_validate(image))


@router.get("/", response_model=list[ImageOut])
async def list_images(
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
) -> list[ImageOut]:
    """Return paginated list of uploaded images."""
    result = await db.execute(select(Image).offset(skip).limit(limit))
    images = result.scalars().all()
    return [ImageOut.model_validate(img) for img in images]


@router.get("/{image_id}", response_model=ImageOut)
async def get_image(image_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> ImageOut:
    """Return metadata for a single image."""
    image = await db.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return ImageOut.model_validate(image)
