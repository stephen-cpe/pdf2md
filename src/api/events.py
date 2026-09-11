"""Progress event bus — broadcast + replay buffer.

Event schema (§5.2): stage_changed, page_update, page_markdown, log,
job_finished — each a JSON-serializable dict with at least "event" and
"job_id". The hub keeps a bounded per-job replay buffer so a reconnecting
WebSocket resumes the stream where it left off.
"""

import asyncio
from collections import deque
from collections.abc import AsyncGenerator
from typing import Any


class EventBus:
    """One job's broadcast channel with replayable history."""

    def __init__(self, capacity: int = 1000) -> None:
        self._buffer: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    def replay(self) -> list[dict[str, Any]]:
        """Buffered events, oldest first (reconnect catch-up)."""
        return list(self._buffer)

    async def publish(self, event: dict[str, Any]) -> None:
        """Append + fan out (slow subscribers get best-effort delivery)."""
        self._buffer.append(event)
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                continue

    async def subscribe(self) -> AsyncGenerator[dict[str, Any]]:
        """Yield live events until cancelled (caller closes on disconnect)."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)


class BusHub:
    """Per-job buses, created on demand (7.1 app state owns one)."""

    def __init__(self, capacity: int = 1000) -> None:
        self._buses: dict[str, EventBus] = {}
        self._capacity = capacity

    def bus(self, job_id: str) -> EventBus:
        """Existing bus for job_id, creating it on first use."""
        if job_id not in self._buses:
            self._buses[job_id] = EventBus(capacity=self._capacity)
        return self._buses[job_id]

    def drop(self, job_id: str) -> None:
        """Forget a job's bus (history cleanup; safe mid-stream)."""
        self._buses.pop(job_id, None)


__all__ = ["BusHub", "EventBus"]
