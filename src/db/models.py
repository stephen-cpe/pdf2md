"""SQLAlchemy 2.x async models — SRS §6.1.

Tables: jobs, pages, images, events. Verified-page immutability
is enforced at the repository layer, not here.
Provenance hashes ride on jobs (pipeline_version) + pages.
"""

import datetime
import enum
import uuid
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all pdf2md tables."""


class JobStatus(str, enum.Enum):
    """FR-JOB-8 job statuses — stored as native PG enum."""

    QUEUED = "queued"
    PREFLIGHT = "preflight"
    RENDERING = "rendering"
    OCR = "ocr"
    TRANSCRIBING = "transcribing"
    ASSEMBLING = "assembling"
    QA = "qa"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class PageStatus(str, enum.Enum):
    """Per-page statuses (FR-UI-2 status grid)."""

    PENDING = "pending"
    OCR = "ocr"
    TRANSCRIBED = "transcribed"
    VERIFIED = "verified"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class ImageSource(str, enum.Enum):
    """Figure provenance (FR-IMG-1): native PDF object vs rendered crop."""

    NATIVE = "native"
    CROP = "crop"


class Job(Base):
    """One PDF conversion job (SRS §6.1 jobs)."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(512))
    file_sha256: Mapped[str] = mapped_column(String(64))
    page_count: Mapped[int] = mapped_column(default=0)
    status: Mapped[JobStatus] = mapped_column(default=JobStatus.QUEUED)
    stage: Mapped[str] = mapped_column(String(64), default="queued")
    output_dir: Mapped[str] = mapped_column(String(1024))
    options: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    pdf_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    pipeline_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    started_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    pages: Mapped[list[Page]] = relationship(
        back_populates="job", cascade="all, delete-orphan", passive_deletes=True
    )
    images: Mapped[list[Image]] = relationship(
        back_populates="job", cascade="all, delete-orphan", passive_deletes=True
    )
    events: Mapped[list[Event]] = relationship(
        back_populates="job", cascade="all, delete-orphan", passive_deletes=True
    )


class Page(Base):
    """One page checkpoint — the unit of crash recovery (FR-JOB-4)."""

    __tablename__ = "pages"
    __table_args__ = (UniqueConstraint("job_id", "page_number"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int]
    status: Mapped[PageStatus] = mapped_column(default=PageStatus.PENDING)
    render_dpi: Mapped[int | None] = mapped_column(nullable=True)
    ocr_output: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    coverage_score: Mapped[int | None] = mapped_column(nullable=True)
    retries: Mapped[int] = mapped_column(default=0)
    needs_review: Mapped[bool] = mapped_column(default=False)
    omissions: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    token_usage: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    timings: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    source_page_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    render_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ocr_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    markdown_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    job: Mapped[Job] = relationship(back_populates="pages")


class Image(Base):
    """One extracted figure asset (SRS §6.1 images)."""

    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int]
    asset_path: Mapped[str] = mapped_column(String(1024))
    source: Mapped[ImageSource]
    bbox: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    alt_text: Mapped[str | None] = mapped_column(String(512), nullable=True)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    referenced: Mapped[bool] = mapped_column(default=False)

    job: Mapped[Job] = relationship(back_populates="images")


class Event(Base):
    """Audit/log trail per job."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    ts: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    level: Mapped[str] = mapped_column(String(16))
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message: Mapped[str] = mapped_column(Text)

    job: Mapped[Job] = relationship(back_populates="events")


__all__ = ["Base", "Event", "Image", "ImageSource", "Job", "JobStatus", "Page", "PageStatus"]
