"""scripted 429/500/timeout sequences (no network)."""

import httpx
import pytest

from src.pipeline.resilience import PauseJob, resilient


def _status_error(status: int, retry_after: str | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://ollama.com/api/chat")
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = httpx.Response(status, headers=headers, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


async def _no_sleep(delay: float) -> None:
    """Sleeper stub: records nothing, waits nothing."""
    _ = delay


def _scripted(outcomes: list, results: list | None = None):
    """Async callable playing a script: Exception instances raise, else return."""

    async def _call():
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return _call


async def test_success_no_sleep() -> None:
    delays: list[float] = []

    async def _sleep(delay: float) -> None:
        delays.append(delay)

    assert await resilient(_scripted(["ok"]), sleeper=_sleep) == "ok"
    assert delays == []


async def test_500_then_success_backs_off() -> None:
    delays: list[float] = []

    async def _sleep(delay: float) -> None:
        delays.append(delay)

    result = await resilient(
        _scripted([_status_error(500), _status_error(503), "recovered"]),
        base_delay=0.5,
        sleeper=_sleep,
    )
    assert result == "recovered"
    assert delays == [0.5, 1.0]


async def test_429_honors_retry_after() -> None:
    delays: list[float] = []

    async def _sleep(delay: float) -> None:
        delays.append(delay)

    result = await resilient(
        _scripted([_status_error(429, "0.25"), "recovered"]),
        base_delay=10.0,
        sleeper=_sleep,
    )
    assert result == "recovered"
    assert delays == [0.25]


async def test_sustained_500_pauses() -> None:
    calls = 0

    async def _call():
        nonlocal calls
        calls += 1
        raise _status_error(500)

    with pytest.raises(PauseJob, match="3 attempts"):
        await resilient(_call, sleeper=_no_sleep)
    assert calls == 3


async def test_sustained_timeout_pauses() -> None:
    calls = 0

    async def _call():
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(PauseJob, match="ocr call"):
        await resilient(_call, operation="ocr call", sleeper=_no_sleep)
    assert calls == 3


async def test_400_raises_immediately() -> None:
    calls = 0
    delays: list[float] = []

    async def _call():
        nonlocal calls
        calls += 1
        raise _status_error(400)

    async def _sleep(delay: float) -> None:
        delays.append(delay)

    with pytest.raises(httpx.HTTPStatusError):
        await resilient(_call, sleeper=_sleep)
    assert calls == 1 and delays == []


async def test_connect_error_retries() -> None:
    delays: list[float] = []

    async def _sleep(delay: float) -> None:
        delays.append(delay)

    result = await resilient(
        _scripted([httpx.ConnectError("dns"), "recovered"]), base_delay=0.25, sleeper=_sleep
    )
    assert result == "recovered"
    assert delays == [0.25]


async def test_non_http_error_propagates() -> None:
    async def _call():
        raise ValueError("caller bug")

    with pytest.raises(ValueError, match="caller bug"):
        await resilient(_call)


async def test_retries_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from src.logging import configure_logging

    configure_logging()
    root = logging.getLogger("pdf2md")
    root.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="pdf2md"):
            await resilient(
                _scripted([_status_error(500), "recovered"]),
                base_delay=0.01,
                sleeper=_no_sleep,
            )
    finally:
        root.removeHandler(caplog.handler)
    warnings = [r.getMessage() for r in caplog.records if r.name.startswith("pdf2md.")]
    assert any("attempt 1/3" in message for message in warnings)
