"""Driver 8.2 paths: unwritable output pauses (not fails). Real DB, fake models."""

import uuid
from pathlib import Path

import pymupdf
import pytest

from src.config import load_settings
from src.db import engine as engine_mod
from src.db import repo
from src.db.models import Job, JobStatus
from src.pipeline import driver as driver_mod
from src.pipeline.agent import AgentResult
from src.pipeline.control import JobControl
from src.pipeline.driver import JobOptions, run_job
from src.pipeline.ocr import OcrResult
from src.workspace import Workspace

pytestmark = pytest.mark.integration


def _env(markdown: str) -> str:
    return (
        "<<<MARKDOWN>>>\n" + markdown + "\n<<<END_MARKDOWN>>>\n"
        "<<<FIGURES>>>\n[]\n<<<END_FIGURES>>>\n"
        "<<<FURNITURE>>>\n{}\n<<<END_FURNITURE>>>\n<<<NOTES>>>\n-\n<<<END_NOTES>>>"
    )


def _verdict() -> str:
    return '<<<VERDICT>>>\n{"coverage": 99, "misses": [], "structure_issues": [], "verdict": "pass"}\n<<<END_VERDICT>>>'


class _Ocr:
    async def run_page(self, image: Path, prompt: str = "") -> OcrResult:
        return OcrResult(text="ocr text", prompt=prompt, model="fake", latency_ms=1.0)


class _Agent:
    def __init__(self, raws: dict[int, list[str]]):
        self.raws = {page: list(items) for page, items in raws.items()}

    async def transcribe(self, image, ocr_text, w, h, n=1, ctx="", outline=""):
        items = self.raws.get(n, [_env("# fallback")])
        raw = items.pop(0) if len(items) > 1 else items[0]
        return AgentResult(raw, "fake", "low", 1.0, 5, 5)


def _pdf(path: Path) -> Path:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "driver e2e one " * 20)
    doc.new_page().insert_text((72, 72), "driver e2e two " * 20)
    doc.save(path)
    doc.close()
    return path


async def test_unwritable_output_pauses_job(tmp_path: Path, monkeypatch) -> None:
    settings = load_settings()
    dsn = settings.DATABASE_URL.get_secret_value()
    engine = engine_mod.create_engine(dsn)
    factory = engine_mod.session_factory(engine)
    created: list = []
    try:
        pdf = _pdf(tmp_path / "doc.pdf")
        out = tmp_path / "out"
        out.mkdir()
        async with factory() as session:
            job = await repo.create_job(
                session,
                filename=f"drv-{uuid.uuid4().hex}.pdf",
                file_sha256="j" * 64,
                page_count=2,
                output_dir=str(out),
                options={},
            )
            created.append(job.id)
            jid = job.id
        options = JobOptions(
            render_dpi=200, coverage_threshold=90, max_page_retries=1, rolling_context_pages=2
        )
        deps_patch = {
            "ocr": _Ocr(),
            "agent": _Agent({1: [_env("# One")], 2: [_env("# Two")]}),
            "verifier": _Agent({1: [_verdict()], 2: [_verdict()]}),
        }
        real_page_deps = driver_mod.PageDeps
        fakes = dict(deps_patch)

        def _fake_deps(**kwargs):
            _ = kwargs  # driver-passed real adapters intentionally ignored
            return real_page_deps(
                ocr=fakes["ocr"],
                agent=fakes["agent"],
                verifier=fakes["verifier"],
                render_dpi=options.render_dpi,
                coverage_threshold=options.coverage_threshold,
                max_page_retries=options.max_page_retries,
                rolling_context_pages=options.rolling_context_pages,
            )

        monkeypatch.setattr(driver_mod, "PageDeps", _fake_deps)

        def _boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(driver_mod, "write_output", _boom)
        emitted: list[dict] = []

        async def _emit(event: dict) -> None:
            emitted.append(event)

        workspace = Workspace(tmp_path / "ws")
        result = await run_job(
            factory=factory,
            settings=settings,
            workspace=workspace,
            job_id=jid,
            pdf_path=pdf,
            output_dir=out,
            options=options,
            control=JobControl(),
            emit=_emit,
        )
        assert result == "paused"
        async with factory() as session:
            job = await repo.get_job(session, jid)
            assert job is not None and job.status is JobStatus.PAUSED
            assert await repo.find_resume_point(session, jid) is None  # all pages done
        assert emitted[-1]["status"] == "paused"
    finally:
        async with factory() as session:
            for job_id in created:
                job = await session.get(Job, job_id)
                if job is not None:
                    await session.delete(job)
            await session.commit()
        await engine.dispose()
