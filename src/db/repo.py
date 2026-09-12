"""Job/page repository — atomic checkpoints, resume, history.

Conventions: page_number is 1-based (matches UI "page N of M" and
page-{n:03d} asset names). Every mutating function commits exactly one
transaction (FR-JOB-4 checkpoint = the pages-row commit).

A page in verified/needs_review is immutable — checkpoint_page
refuses to touch it (VerifiedPageImmutableError); only the explicit
invalidate_page op reopens it. Resume therefore never re-runs terminal pages.
"""

import uuid
from typing import Any

from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Event, Image, ImageSource, Job, JobStatus, Page, PageStatus

TERMINAL_PAGE_STATUSES = (PageStatus.VERIFIED, PageStatus.NEEDS_REVIEW)


class VerifiedPageImmutableError(RuntimeError):
    """Raised when checkpointing a verified/needs_review page implicitly."""


async def create_job(
    session: AsyncSession,
    *,
    filename: str,
    file_sha256: str,
    page_count: int,
    output_dir: str,
    options: dict[str, Any] | None = None,
    pdf_metadata: dict[str, Any] | None = None,
    pipeline_version: str | None = None,
    job_id: uuid.UUID | None = None,
) -> Job:
    """Insert a queued job; one transaction (caller-supplied id optional)."""
    job = Job(
        id=job_id or uuid.uuid4(),
        filename=filename,
        file_sha256=file_sha256,
        page_count=page_count,
        output_dir=output_dir,
        options=options or {},
        pdf_metadata=pdf_metadata or {},
        pipeline_version=pipeline_version,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def get_job(session: AsyncSession, job_id: uuid.UUID) -> Job | None:
    """Fetch one job by id (None when unknown)."""
    return await session.get(Job, job_id)


async def set_job_status(
    session: AsyncSession, job_id: uuid.UUID, status: JobStatus, stage: str | None = None
) -> None:
    """Update job status (+ optional stage); one transaction."""
    job = await session.get(Job, job_id)
    if job is None:
        raise KeyError(f"unknown job {job_id}")
    job.status = status
    if stage is not None:
        job.stage = stage
    await session.commit()


async def checkpoint_page(
    session: AsyncSession,
    job_id: uuid.UUID,
    page_number: int,
    *,
    event_message: str | None = None,
    event_level: str = "info",
    **fields: Any,
) -> Page:
    """Upsert one page row + optional audit event in ONE transaction.

    Raises VerifiedPageImmutableError when the existing row is terminal
    — call invalidate_page first for an explicit re-run.
    """
    existing = await session.scalar(
        select(Page).where(Page.job_id == job_id, Page.page_number == page_number)
    )
    if existing is not None and existing.status in TERMINAL_PAGE_STATUSES:
        raise VerifiedPageImmutableError(
            f"page {page_number} of job {job_id} is {existing.status.value}: immutable"
        )
    page = existing or Page(job_id=job_id, page_number=page_number)
    for key, value in fields.items():
        setattr(page, key, value)
    session.add(page)
    if event_message is not None:
        session.add(Event(job_id=job_id, level=event_level, message=event_message))
    await session.commit()
    await session.refresh(page)
    # Post-commit readback: a "verified" log must never outlive a missing row.
    # Any storage anomaly surfaces here loudly instead of as a phantom page.
    check = await session.scalar(
        select(Page).where(Page.job_id == job_id, Page.page_number == page_number)
    )
    if check is None or check.id != page.id:
        raise RuntimeError(f"checkpoint for page {page_number} of job {job_id} did not persist")
    return page


async def invalidate_page(session: AsyncSession, job_id: uuid.UUID, page_number: int) -> None:
    """Explicit recovery op: reopen a page for re-transcription."""
    page = await session.scalar(
        select(Page).where(Page.job_id == job_id, Page.page_number == page_number)
    )
    if page is None:
        raise KeyError(f"unknown page {page_number} of job {job_id}")
    page.status = PageStatus.PENDING
    page.needs_review = False
    session.add(
        Event(job_id=job_id, level="warning", message=f"page {page_number} invalidated for re-run")
    )
    await session.commit()


async def find_resume_point(session: AsyncSession, job_id: uuid.UUID) -> int | None:
    """First 1-based page number not terminal; None when the job is complete."""
    job = await session.get(Job, job_id)
    if job is None:
        raise KeyError(f"unknown job {job_id}")
    terminal = set(
        (
            await session.scalars(
                select(Page.page_number).where(
                    Page.job_id == job_id, Page.status.in_(TERMINAL_PAGE_STATUSES)
                )
            )
        ).all()
    )
    for number in range(1, job.page_count + 1):
        if number not in terminal:
            return number
    return None


async def log_event(
    session: AsyncSession, job_id: uuid.UUID, level: str, message: str, stage: str | None = None
) -> None:
    """Append one audit event; one transaction."""
    session.add(Event(job_id=job_id, level=level, stage=stage, message=message))
    await session.commit()


async def record_image(
    session: AsyncSession,
    *,
    job_id: uuid.UUID,
    page_number: int,
    asset_path: str,
    source: ImageSource,
    bbox: dict[str, Any] | None = None,
    alt_text: str | None = None,
    caption: str | None = None,
    referenced: bool = True,
    mermaid: str | None = None,
    diagram_type: str | None = None,
    conversion_status: str | None = None,
    confidence: int | None = None,
) -> Image:
    """Record one extracted figure asset; caller commits via checkpoint flow."""
    image = Image(
        job_id=job_id,
        page_number=page_number,
        asset_path=asset_path,
        source=source,
        bbox=bbox,
        alt_text=alt_text,
        caption=caption,
        referenced=referenced,
        mermaid=mermaid,
        diagram_type=diagram_type,
        conversion_status=conversion_status,
        confidence=confidence,
    )
    session.add(image)
    await session.flush()
    return image


async def clear_job_images(session: AsyncSession, job_id: uuid.UUID) -> int:
    """Delete a job's image rows (assembly regenerates them; keeps restarts idempotent)."""
    from sqlalchemy import func

    remaining = await session.scalar(
        select(func.count()).select_from(Image).where(Image.job_id == job_id)
    )
    await session.execute(delete(Image).where(Image.job_id == job_id))
    return remaining or 0


async def job_history(session: AsyncSession, limit: int = 50) -> list[Job]:
    """Newest-first job list (FR-JOB-7)."""
    result = await session.scalars(select(Job).order_by(desc(Job.created_at)).limit(limit))
    return list(result.all())


__all__ = [
    "TERMINAL_PAGE_STATUSES",
    "VerifiedPageImmutableError",
    "checkpoint_page",
    "clear_job_images",
    "create_job",
    "find_resume_point",
    "get_job",
    "invalidate_page",
    "job_history",
    "log_event",
    "record_image",
    "set_job_status",
]
