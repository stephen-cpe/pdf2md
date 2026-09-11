"""Real DB: crash → resume point; one-transaction checkpoint.

Integration tier (NFR-11): real Postgres via .env, isolated by uuid job
filenames + cascade cleanup. Cloud model always mocked (none used here).
"""

import asyncio
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError

from src.config import load_settings
from src.db import engine as engine_mod
from src.db import repo
from src.db.models import Event, Job, Page, PageStatus

pytestmark = pytest.mark.integration


@pytest.fixture()
async def sessions():
    """Per-test engine (one event loop each) + cascade cleanup of created jobs."""
    dsn = load_settings().DATABASE_URL.get_secret_value()
    engine = engine_mod.create_engine(dsn)
    factory = engine_mod.session_factory(engine)
    created: list[uuid.UUID] = []
    yield factory, created
    async with factory() as session:
        for job_id in created:
            job = await session.get(Job, job_id)
            if job is not None:
                await session.delete(job)
        await session.commit()
    await engine.dispose()


def _name(tag: str) -> str:
    return f"test-1.2-{tag}-{uuid.uuid4().hex}.pdf"


async def test_crash_resume_point(sessions) -> None:
    """Pages 1-3 verified, 'crash' (sessions dropped), resume lands on 4."""
    make_session, created = sessions
    async with make_session() as session:
        job = await repo.create_job(
            session,
            filename=_name("crash"),
            file_sha256="a" * 64,
            page_count=5,
            output_dir=".",
        )
        created.append(job.id)
        for number in (1, 2, 3):
            await repo.checkpoint_page(
                session, job.id, number, status=PageStatus.VERIFIED, markdown=f"p{number}"
            )
    # Crash: all sessions gone. Fresh session resumes.
    async with make_session() as session:
        assert await repo.find_resume_point(session, job.id) == 4
        # Non-terminal progress does not move the pointer past unfinished work.
        await repo.checkpoint_page(session, job.id, 4, status=PageStatus.TRANSCRIBED)
        assert await repo.find_resume_point(session, job.id) == 4
        await repo.checkpoint_page(session, job.id, 4, status=PageStatus.VERIFIED)
        await repo.checkpoint_page(session, job.id, 5, status=PageStatus.NEEDS_REVIEW)
        assert await repo.find_resume_point(session, job.id) is None


async def test_checkpoint_is_one_transaction(sessions) -> None:
    """Page row + audit event commit together: a mid-commit failure leaves neither."""
    make_session, created = sessions
    async with make_session() as session:
        job = await repo.create_job(
            session, filename=_name("atomic"), file_sha256="b" * 64, page_count=2, output_dir="."
        )
        created.append(job.id)
        jid = job.id
        ok = await repo.checkpoint_page(
            session, job.id, 1, status=PageStatus.VERIFIED, event_message="page 1 done"
        )
        assert ok.page_number == 1
        events = (await session.scalars(select(Event).where(Event.job_id == jid))).all()
        assert len(events) == 1
        # Event level over varchar(16) fails at the DB: page row must roll back too.
        with pytest.raises(DBAPIError):
            await repo.checkpoint_page(
                session,
                jid,
                2,
                status=PageStatus.VERIFIED,
                event_message="page 2 done",
                event_level="L" * 20,
            )
        await session.rollback()
        missing = await session.scalar(
            select(Page).where(Page.job_id == jid, Page.page_number == 2)
        )
        assert missing is None


async def test_verified_page_immutable_until_invalidated(sessions) -> None:
    """Implicit overwrite refused; explicit invalidate reopens."""
    make_session, created = sessions
    async with make_session() as session:
        job = await repo.create_job(
            session, filename=_name("immut"), file_sha256="c" * 64, page_count=1, output_dir="."
        )
        created.append(job.id)
        jid = job.id
        await repo.checkpoint_page(session, jid, 1, status=PageStatus.VERIFIED, markdown="v1")
        with pytest.raises(repo.VerifiedPageImmutableError):
            await repo.checkpoint_page(session, jid, 1, markdown="v2")
        await session.rollback()
        row = await session.scalar(select(Page).where(Page.job_id == jid, Page.page_number == 1))
        assert row is not None and row.markdown == "v1"
        await repo.invalidate_page(session, jid, 1)
        reopened = await repo.checkpoint_page(session, jid, 1, markdown="v2")
        assert reopened.markdown == "v2"


async def test_job_history_newest_first(sessions) -> None:
    make_session, created = sessions
    names: list[str] = []
    async with make_session() as session:
        for tag in ("h1", "h2", "h3"):
            name = _name(tag)
            names.append(name)
            job = await repo.create_job(
                session, filename=name, file_sha256="d" * 64, page_count=1, output_dir="."
            )
            created.append(job.id)
            await asyncio.sleep(0.02)
    async with make_session() as session:
        history = await repo.job_history(session, limit=2)
        assert [j.filename for j in history] == [names[2], names[1]]
