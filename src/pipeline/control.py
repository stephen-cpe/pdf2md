"""Job control plane — status machine + pause/cancel signals.

FR-JOB-8 statuses with an explicit transition map: illegal transitions raise
instead of silently corrupting job state. Pause/cancel are asyncio-level
signals honored BETWEEN pages — a page in flight always finishes
its checkpoint first, so cancel mid-page still leaves resumable state.
Terminal states (completed/failed/cancelled) have no exits; a new job is
required (different runs are different jobs).
"""

import asyncio

from src.db.models import JobStatus

_ACTIVE = (
    JobStatus.PREFLIGHT,
    JobStatus.RENDERING,
    JobStatus.OCR,
    JobStatus.TRANSCRIBING,
    JobStatus.ASSEMBLING,
    JobStatus.QA,
)

TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset({JobStatus.PREFLIGHT, JobStatus.PAUSED, JobStatus.CANCELLED}),
    JobStatus.PREFLIGHT: frozenset({JobStatus.RENDERING, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.RENDERING: frozenset(
        {JobStatus.OCR, JobStatus.PAUSED, JobStatus.FAILED, JobStatus.CANCELLED}
    ),
    JobStatus.OCR: frozenset(
        {JobStatus.TRANSCRIBING, JobStatus.PAUSED, JobStatus.FAILED, JobStatus.CANCELLED}
    ),
    JobStatus.TRANSCRIBING: frozenset(
        {JobStatus.ASSEMBLING, JobStatus.PAUSED, JobStatus.FAILED, JobStatus.CANCELLED}
    ),
    JobStatus.ASSEMBLING: frozenset(
        {JobStatus.QA, JobStatus.PAUSED, JobStatus.FAILED, JobStatus.CANCELLED}
    ),
    JobStatus.QA: frozenset(
        {JobStatus.COMPLETED, JobStatus.PAUSED, JobStatus.FAILED, JobStatus.CANCELLED}
    ),
    JobStatus.PAUSED: frozenset({JobStatus.QUEUED, *_ACTIVE, JobStatus.CANCELLED}),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}


class IllegalTransitionError(RuntimeError):
    """Status change outside the FR-JOB-8 map."""


class JobControl:
    """In-memory control for one job run (persisted via repo by the caller)."""

    def __init__(self, status: JobStatus = JobStatus.QUEUED) -> None:
        self._status = status
        self._resume = asyncio.Event()
        self._resume.set()
        self._cancelled = False
        self._paused_from: JobStatus | None = None

    @property
    def status(self) -> JobStatus:
        """Current status."""
        return self._status

    @property
    def cancel_requested(self) -> bool:
        """True after request_cancel until consumed by the loop."""
        return self._cancelled

    def transition(self, target: JobStatus) -> None:
        """Move status; raises IllegalTransitionError off-map."""
        if target not in TRANSITIONS[self._status]:
            raise IllegalTransitionError(f"{self._status.value} -> {target.value} forbidden")
        self._status = target

    def request_pause(self) -> None:
        """Pause between pages (only from an active stage)."""
        self._paused_from = self._status
        self.transition(JobStatus.PAUSED)
        self._resume.clear()

    def request_resume(self, target: JobStatus | None = None) -> None:
        """Resume a paused job into its pre-pause stage (or an explicit one)."""
        if self._status is not JobStatus.PAUSED:
            raise IllegalTransitionError(f"resume requires PAUSED, in {self._status.value}")
        self.transition(target or self._paused_from or JobStatus.TRANSCRIBING)
        self._paused_from = None
        self._resume.set()

    def request_cancel(self) -> None:
        """Cancel from any non-terminal status; unblocks a paused waiter."""
        if self._status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            raise IllegalTransitionError(f"cannot cancel terminal {self._status.value}")
        self._status = JobStatus.CANCELLED
        self._cancelled = True
        self._resume.set()

    async def wait_if_paused(self) -> None:
        """Block while paused (returns immediately otherwise)."""
        await self._resume.wait()


__all__ = ["TRANSITIONS", "IllegalTransitionError", "JobControl"]
