"""Job model — tracks async Celery image-processing tasks."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.core.database import Base


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class JobType(StrEnum):
    EMBED_WATERMARK = "embed_watermark"
    EXTRACT_WATERMARK = "extract_watermark"
    RECOVER_WATERMARK = "recover_watermark"
    DETECT_FORGERY = "detect_forgery"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    celery_task_id: Mapped[str | None] = mapped_column(String(255), index=True)
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default=JobStatus.PENDING)
    result_path: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # FK → images
    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("images.id", ondelete="CASCADE"), nullable=False
    )
    image: Mapped[Image] = relationship(back_populates="jobs")  # noqa: F821

    def __repr__(self) -> str:
        return f"<Job {self.id} type={self.job_type} status={self.status}>"
