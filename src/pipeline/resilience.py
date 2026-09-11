"""Shared resilience wrapper — retries then pause, never fail.

FR-AGT-8: every model call gets timeout (at the client) + exponential-backoff
retries (default 3 attempts) + 429/Retry-After respect. Sustained outage
raises PauseJob — the pipeline pauses the job (resumable), it never fails it.
4xx (other than 429) and non-transport errors propagate immediately.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

Sleeper = Callable[[float], Awaitable[None]]

#: Hard ceiling for a single Retry-After sleep (a malicious huge header
#: must not wedge the worker; the next attempt re-reads the header anyway).
MAX_RETRY_AFTER_SECONDS = 120.0


class PauseJob(RuntimeError):
    """Sustained outage — pause the job (resumable), do not fail it."""


def _retry_delay(exc: BaseException, attempt: int, base_delay: float) -> float | None:
    """Seconds to wait before the next attempt, or None when not retryable."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            header = exc.response.headers.get("retry-after")
            if header is not None:
                try:
                    return min(float(header), MAX_RETRY_AFTER_SECONDS)
                except ValueError:
                    pass  # unparsable date form: fall through to backoff
            return base_delay * (2.0 ** (attempt - 1))
        if status >= 500:
            return base_delay * (2.0 ** (attempt - 1))
        return None
    if isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            TimeoutError,
            asyncio.TimeoutError,
        ),
    ):
        return base_delay * (2.0 ** (attempt - 1))
    return None


async def resilient[T](
    call: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    operation: str = "model call",
    sleeper: Sleeper = asyncio.sleep,
    extra: dict[str, Any] | None = None,
) -> T:
    """Run call with backoff; raise PauseJob when attempts run out.

    Only httpx transport/5xx/429 failures retry — everything else (4xx,
    validation, cancellation) propagates untouched. Each retry is logged
    (incident 5fbf: an 11-minute silent retry window taught us operators
    need to see backoff happening, not just its outcome).
    """
    from src.logging import get_logger

    logger = get_logger("resilience")
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    context = {"operation": operation, **(extra or {})}
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except (httpx.HTTPError, TimeoutError) as exc:
            delay = _retry_delay(exc, attempt, base_delay)
            if delay is None:
                # Diagnostic: non-retryable transport errors (e.g. ReadError
                # 10053) previously propagated silently; log operation + type
                # so the next failure pinpoints ocr vs transcribe vs verify.
                logger.warning(
                    "%s non-retryable (%s: %s); propagating",
                    operation,
                    type(exc).__name__,
                    exc,
                    extra=context,
                )
                raise
            last_error = exc
            if attempt >= attempts:
                break
            logger.warning(
                "%s attempt %d/%d failed (%s); retrying in %.1fs",
                operation,
                attempt,
                attempts,
                exc,
                delay,
                extra=context,
            )
            await sleeper(delay)
    assert last_error is not None
    raise PauseJob(f"{operation} failed after {attempts} attempts: {last_error}") from last_error


__all__ = ["MAX_RETRY_AFTER_SECONDS", "PauseJob", "resilient"]
