"""EventBus shutdown hygiene: cancelled subscribers clean up (Ctrl+C quiet)."""

import asyncio

from src.api.events import BusHub


async def test_subscribe_cancel_removes_subscriber() -> None:
    bus = BusHub().bus("job-1")
    seen: list[dict] = []

    async def _consume() -> None:
        async for event in bus.subscribe():
            seen.append(event)

    task = asyncio.create_task(_consume())
    await asyncio.sleep(0)
    assert len(bus._subscribers) == 1
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert bus._subscribers == set()
    assert seen == []


async def test_publish_replay_then_cancel() -> None:
    bus = BusHub().bus("job-2")
    await bus.publish({"event": "log", "job_id": "job-2", "message": "m0"})
    assert [e["message"] for e in bus.replay()] == ["m0"]

    received: list[dict] = []

    async def _consume() -> None:
        async for event in bus.subscribe():
            received.append(event)

    task = asyncio.create_task(_consume())
    await asyncio.sleep(0)
    await bus.publish({"event": "log", "job_id": "job-2", "message": "m1"})
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert received == [{"event": "log", "job_id": "job-2", "message": "m1"}]
    assert bus._subscribers == set()
