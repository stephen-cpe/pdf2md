"""Unit tier: transition map enforced, signals behave (no DB)."""

import asyncio

import pytest

from src.db.models import JobStatus
from src.pipeline.control import IllegalTransitionError, JobControl


def test_happy_chain_to_completed() -> None:
    control = JobControl()
    for target in (
        JobStatus.PREFLIGHT,
        JobStatus.RENDERING,
        JobStatus.OCR,
        JobStatus.TRANSCRIBING,
        JobStatus.ASSEMBLING,
        JobStatus.QA,
        JobStatus.COMPLETED,
    ):
        control.transition(target)
    assert control.status is JobStatus.COMPLETED


@pytest.mark.parametrize(
    ("from_status", "to_status"),
    [
        (JobStatus.QUEUED, JobStatus.TRANSCRIBING),
        (JobStatus.QUEUED, JobStatus.COMPLETED),
        (JobStatus.TRANSCRIBING, JobStatus.QUEUED),
        (JobStatus.TRANSCRIBING, JobStatus.COMPLETED),
        (JobStatus.COMPLETED, JobStatus.QUEUED),
        (JobStatus.FAILED, JobStatus.QUEUED),
        (JobStatus.CANCELLED, JobStatus.TRANSCRIBING),
        (JobStatus.PAUSED, JobStatus.COMPLETED),
    ],
)
def test_illegal_transitions_raise(from_status: JobStatus, to_status: JobStatus) -> None:
    control = JobControl(from_status)
    with pytest.raises(IllegalTransitionError):
        control.transition(to_status)
    assert control.status is from_status


def test_pause_resume_roundtrip() -> None:
    control = JobControl(JobStatus.TRANSCRIBING)
    control.request_pause()
    assert control.status is JobStatus.PAUSED
    control.request_resume(JobStatus.TRANSCRIBING)
    assert control.status is JobStatus.TRANSCRIBING


def test_pause_from_queued_holds_start() -> None:
    control = JobControl()
    control.request_pause()
    assert control.status is JobStatus.PAUSED
    control.request_resume()
    assert control.status is JobStatus.QUEUED


def test_resume_requires_paused() -> None:
    control = JobControl(JobStatus.TRANSCRIBING)
    with pytest.raises(IllegalTransitionError):
        control.request_resume(JobStatus.TRANSCRIBING)


def test_cancel_from_active() -> None:
    control = JobControl(JobStatus.TRANSCRIBING)
    control.request_cancel()
    assert control.status is JobStatus.CANCELLED
    assert control.cancel_requested


def test_cancel_terminal_raises() -> None:
    with pytest.raises(IllegalTransitionError):
        JobControl(JobStatus.COMPLETED).request_cancel()


async def test_wait_returns_when_running() -> None:
    await asyncio.wait_for(JobControl().wait_if_paused(), timeout=1.0)
