"""JWT authentication dependencies and per-user Redis rate limiter.

Provides FastAPI dependency-injection callables:

- ``get_current_user``  — decodes a Bearer JWT, loads the User from DB.
- ``require_auth``      — alias that raises 401 on failure.
- ``rate_limit``        — per-user sliding-window rate limiter backed by Redis.

Usage in a route::

    @router.post("/embed")
    async def embed(
        body: EmbedRequest,
        user: User = Depends(require_auth),
        _rl: None = Depends(rate_limit()),
    ):
        ...
"""

from __future__ import annotations

import time
from collections.abc import Callable

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import get_settings
from backend.app.core.database import get_db
from backend.app.core.security import decode_access_token
from backend.app.models.user import User

settings = get_settings()

# ---------------------------------------------------------------------------
# Bearer token extractor
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Redis connection pool (lazy singleton)
# ---------------------------------------------------------------------------

_redis_pool: aioredis.Redis | None = None


async def _get_redis() -> aioredis.Redis:
    """Return (or create) the async Redis connection pool."""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
    return _redis_pool


# ---------------------------------------------------------------------------
# Auth dependency — get_current_user
# ---------------------------------------------------------------------------

async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Decode the JWT and return the authenticated ``User``.

    Raises ``401 Unauthorized`` if the token is missing, invalid, or the
    referenced user does not exist.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: str | None = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing 'sub' claim",
        )

    user = await db.execute(select(User).where(User.id == user_id))
    user_obj = user.scalars().first()
    if user_obj is None or not user_obj.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    return user_obj


# Convenience alias
require_auth = get_current_user


# ---------------------------------------------------------------------------
# Rate limiter dependency (factory)
# ---------------------------------------------------------------------------

def rate_limit(
    max_requests: int = 30,
    window_seconds: int = 60,
) -> Callable:
    """Return a FastAPI dependency that enforces a per-user sliding-window
    rate limit using Redis sorted sets.

    Parameters
    ----------
    max_requests : int
        Maximum allowed requests within the sliding window.
    window_seconds : int
        Length of the sliding window in seconds.

    Returns
    -------
    An async callable suitable for ``Depends()``.

    Raises
    ------
    HTTPException(429)
        When the rate limit is exceeded.
    """

    async def _check_rate_limit(
        request: Request,
        user: User = Depends(require_auth),
    ) -> None:
        r = await _get_redis()
        key = f"rl:{user.id}:{request.url.path}"
        now = time.time()
        window_start = now - window_seconds

        pipe = r.pipeline()
        # Remove entries older than the window
        pipe.zremrangebyscore(key, 0, window_start)
        # Add the current timestamp
        pipe.zadd(key, {str(now): now})
        # Count entries in the window
        pipe.zcard(key)
        # Auto-expire the whole key after the window
        pipe.expire(key, window_seconds + 1)
        results = await pipe.execute()

        current_count: int = results[2]

        if current_count > max_requests:
            retry_after = window_seconds
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Max {max_requests} requests per {window_seconds}s.",
                headers={"Retry-After": str(retry_after)},
            )

    return _check_rate_limit


# ---------------------------------------------------------------------------
# Auth route — token issuance
# ---------------------------------------------------------------------------

async def authenticate_user(
    email: str,
    password: str,
    db: AsyncSession,
) -> User | None:
    """Verify credentials and return the User or ``None``."""
    from backend.app.core.security import verify_password

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalars().first()
    if user is None:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user
