"""FastAPI application entry-point."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from backend.app.core.config import get_settings
from backend.app.core.database import engine, Base, async_session_factory
from backend.app.models.user import User
from backend.app.routes import auth, health, images, jobs, watermark

settings = get_settings()

_PLACEHOLDER_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    """Startup / shutdown lifecycle hook."""
    # Create tables if they don't exist (for local dev; use Alembic in prod)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    # Seed a placeholder dev user so image uploads don't hit a FK violation
    async with async_session_factory() as session:
        existing = await session.get(User, _PLACEHOLDER_USER_ID)
        if existing is None:
            session.add(User(
                id=_PLACEHOLDER_USER_ID,
                email="dev@placeholder.local",
                hashed_password="not-a-real-hash",
                full_name="Development User",
            ))
            await session.commit()

    yield
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------- Middleware ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Routers ----------
app.include_router(health.router)
app.include_router(auth.router, prefix=settings.api_v1_prefix)
app.include_router(images.router, prefix=settings.api_v1_prefix)
app.include_router(jobs.router, prefix=settings.api_v1_prefix)
app.include_router(watermark.router, prefix=settings.api_v1_prefix)
