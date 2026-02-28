"""Celery application factory."""

from __future__ import annotations

from celery import Celery

from backend.app.core.config import get_settings

settings = get_settings()


def make_celery() -> Celery:
    """Create and configure the Celery application."""
    celery_app = Celery(
        "watermark_worker",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
    )

    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        # Auto-discover tasks inside backend.app.tasks
        imports=["backend.app.tasks.image_tasks"],
    )

    return celery_app


celery_app = make_celery()
