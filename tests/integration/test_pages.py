"""Integration: real DB + real renders, fake models (no Cloud quota).

Covers 4.1 (transcribe+skip), 4.2 (verify/DPI-bump/cap), 4.3 (resume/pause/
outage), 4.4 (look-ahead equality + overlap), 4.5 (cancel-mid-page resume).
"""

import asyncio
import json
import uuid
from pathlib import Path

import httpx
import pymupdf
import pytest
from sqlalchemy import select

from src.config import load_settings
from src.db import engine as engine_mod
from src.db import repo
from src.db.models import Job, JobStatus, Page, PageStatus
from src.pdf import render_page as real_render
from src.pipeline.agent import AgentResult
from src.pipeline.control import JobControl
from src.pipeline.ocr import OcrResult
from src.pipeline.pages import PageDeps, process_page, run_pages, run_pages_lookahead

pytestmark = pytest.mark.integration


def _env(markdown: str, figures: str = "[]", notes: str = "- n") -> str:
    return (
        "<<<MARKDOWN>>>\n" + markdown + "\n<<<END_MARKDOWN>>>\n"
        "<<<FIGURES>>>\n" + figures + "\n<<<END_FIGURES>>>\n"
        "<<<FURNITURE>>>\n{}\n<<<END_FURNITURE>>>\n"
        "<<<NOTES>>>\n" + notes + "\n<<<END_NOTES>>>"
    )


def _verdict(coverage: int, verdict: str = "pass", misses: list | None = None) -> str:
    return (
        "<<<VERDICT>>>\n"
        + json.dumps(
            {
                "coverage": coverage,
                "misses": misses or [],
                "structure_issues": [],
                "verdict": verdict,
            }
        )
        + "\n<<<END_VERDICT>>>"
    )


class FakeOcr:
    """Canned OCR text; optional failure or delay; records calls."""

    def __init__(self, text: str = "fake ocr", fail: Exception | None = None, delay: float = 0.0):
        self.text = text
        self.fail = fail
        self.delay = delay
        self.calls: list[dict] = []

    async def run_page(self, image: Path, prompt: str = "Text Recognition:") -> OcrResult:
        self.calls.append(
            {"image": str(image), "prompt": prompt, "t_start": asyncio.get_event_loop().time()}
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail is not None:
            raise self.fail
        self.calls[-1]["t_end"] = asyncio.get_event_loop().time()
        return OcrResult(text=self.text, prompt=prompt, model="fake-ocr", latency_ms=1.0)


class FakeAgent:
    """Scripted envelope raws per page; records every call with timestamps."""

    def __init__(self, raws: dict[int, list[str]], hook=None):
        self.raws = {page: list(items) for page, items in raws.items()}
        self.calls: list[dict] = []
        self.hook = hook

    async def transcribe(
        self, image, ocr_text, width, height, page_number=1, rolling_context="", outline=""
    ):
        call = {
            "page": page_number,
            "ocr": ocr_text,
            "dims": (width, height),
            "context": rolling_context,
            "outline": outline,
            "t_start": asyncio.get_event_loop().time(),
        }
        self.calls.append(call)
        if self.hook is not None:
            await self.hook(page_number, call)
        await asyncio.sleep(0.0)
        raws = self.raws.get(page_number, self.raws.get(0, []))
        raw = raws.pop(0) if len(raws) > 1 else (raws[0] if raws else _env("# empty"))
        call["t_end"] = asyncio.get_event_loop().time()
        return AgentResult(raw, "fake", "low", 1.0, 10, 5)


def _pdf(tmp_path: Path, pages: int = 3) -> Path:
    pdf = tmp_path / "doc.pdf"
    doc = pymupdf.open()
    for number in range(pages):
        doc.new_page().insert_text((72, 72), f"page {number + 1} text " * 20)
    doc.save(pdf)
    doc.close()
    return pdf


def _rich_pdf(tmp_path: Path, pages: int = 1) -> Path:
    """Born-digital PDF with a dense text layer (hybrid-routing candidate)."""
    pdf = tmp_path / "rich.pdf"
    doc = pymupdf.open()
    for number in range(pages):
        page = doc.new_page()
        body = " ".join(f"rich{number:02d}word{i:03d}" for i in range(60))
        page.insert_textbox(pymupdf.Rect(72, 72, 540, 720), body, fontsize=11)
    doc.save(pdf)
    doc.close()
    return pdf


@pytest.fixture()
async def db():
    """Per-test engine + factory + cascade cleanup (isolated by uuid names)."""
    dsn = load_settings().DATABASE_URL.get_secret_value()
    engine = engine_mod.create_engine(dsn)
    factory = engine_mod.session_factory(engine)
    created: list = []
    yield factory, created
    async with factory() as session:
        for job_id in created:
            job = await session.get(Job, job_id)
            if job is not None:
                await session.delete(job)
        await session.commit()
    await engine.dispose()


async def _job(factory, created, tmp_path: Path, pages: int = 3):
    async with factory() as session:
        job = await repo.create_job(
            session,
            filename=f"m4-{uuid.uuid4().hex}.pdf",
            file_sha256="f" * 64,
            page_count=pages,
            output_dir=str(tmp_path),
        )
        created.append(job.id)
        return job.id


def _deps(ocr, agent, verifier, **overrides) -> PageDeps:
    args: dict = {"ocr": ocr, "agent": agent, "verifier": verifier}
    args.update(overrides)
    return PageDeps(**args)


# --- 4.1 ---


async def test_transcribe_verified_path(db, tmp_path: Path) -> None:
    factory, created = db
    pdf = _pdf(tmp_path)
    job_id = await _job(factory, created, tmp_path)
    figures = '[{"index": 1, "bbox": [0, 0, 10, 10], "alt": "fig", "caption": null}]'
    agent = FakeAgent({1: [_env("# P1\nbody", figures)]})
    verifier = FakeAgent({1: [_verdict(98)]})
    deps = _deps(FakeOcr(), agent, verifier)
    async with factory() as session:
        outcome = await process_page(
            session,
            job_id=job_id,
            page_number=1,
            pdf_path=pdf,
            renders_dir=tmp_path / "r",
            deps=deps,
        )
    assert outcome.status is PageStatus.VERIFIED and outcome.retries_used == 0
    assert outcome.dpi_used == 200 and outcome.coverage == 98
    async with factory() as session:
        row = await session.scalar(select(Page).where(Page.job_id == job_id))
        assert row is not None and row.markdown == "# P1\nbody"
        assert row.coverage_score == 98 and row.render_dpi == 200
        assert row.omissions["figures"][0]["alt"] == "fig"
        assert row.token_usage == {"prompt": 20, "completion": 10}
        assert row.timings["image_width"] > 0 and row.timings["image_height"] > 0
        assert row.timings["model"] == "fake" and row.timings["thinking_effort"] == "low"
        assert row.timings["verification_score"] == 98
        assert row.timings["image_bytes"] and row.timings["image_bytes"] > 0
        assert row.source_page_hash and row.render_hash and row.ocr_hash and row.markdown_hash
        from src.pipeline.hashes import page_source_hash

        assert row.source_page_hash == page_source_hash(pdf, 1)


async def test_terminal_page_skipped_without_calls(db, tmp_path: Path) -> None:
    factory, created = db
    pdf = _pdf(tmp_path, pages=1)
    job_id = await _job(factory, created, tmp_path, pages=1)
    async with factory() as session:
        await repo.checkpoint_page(session, job_id, 1, status=PageStatus.VERIFIED, markdown="v")

    async def _boom(*a, **k):
        raise AssertionError("model must not be called for terminal pages")

    deps = _deps(_boom, _boom, _boom)
    async with factory() as session:
        outcome = await process_page(
            session,
            job_id=job_id,
            page_number=1,
            pdf_path=pdf,
            renders_dir=tmp_path / "r",
            deps=deps,
        )
    assert outcome.skipped and outcome.markdown == "v"


# --- 4.2 ---


async def test_needs_review_after_cap_with_dpi_bump(db, tmp_path: Path) -> None:
    factory, created = db
    pdf = _pdf(tmp_path, pages=1)
    job_id = await _job(factory, created, tmp_path, pages=1)
    renders = tmp_path / "r"
    seen_dpis: list[int] = []
    real = real_render

    def _render(pdf_path, page, dpi, out):
        seen_dpis.append(dpi)
        return real(pdf_path, page, dpi, out)

    agent = FakeAgent({1: [_env("# P1"), _env("# P1"), _env("# P1")]})
    verifier = FakeAgent({1: [_verdict(50, "retry", ["m"])] * 3})
    deps = _deps(FakeOcr(), agent, verifier, max_page_retries=2)
    deps.render_fn = _render
    async with factory() as session:
        outcome = await process_page(
            session, job_id=job_id, page_number=1, pdf_path=pdf, renders_dir=renders, deps=deps
        )
    assert outcome.status is PageStatus.NEEDS_REVIEW
    assert outcome.retries_used == 2 and outcome.coverage == 50
    assert len(agent.calls) == 3  # exactly 1 + MAX_PAGE_RETRIES
    assert seen_dpis == [200, 300, 400]
    assert outcome.dpi_used == 400


async def test_malformed_verdict_counts_down(db, tmp_path: Path) -> None:
    factory, created = db
    pdf = _pdf(tmp_path, pages=1)
    job_id = await _job(factory, created, tmp_path, pages=1)
    agent = FakeAgent({1: [_env("# P1")]})
    verifier = FakeAgent({1: ["garbage"]})
    deps = _deps(FakeOcr(), agent, verifier, max_page_retries=1)
    async with factory() as session:
        outcome = await process_page(
            session,
            job_id=job_id,
            page_number=1,
            pdf_path=pdf,
            renders_dir=tmp_path / "r",
            deps=deps,
        )
    assert outcome.status is PageStatus.NEEDS_REVIEW and outcome.coverage == 0


async def test_malformed_envelope_corrective_then_ok(db, tmp_path: Path) -> None:
    factory, created = db
    pdf = _pdf(tmp_path, pages=1)
    job_id = await _job(factory, created, tmp_path, pages=1)
    agent = FakeAgent({1: ["garbage", _env("# P1 fixed")]})
    verifier = FakeAgent({1: [_verdict(96)]})
    deps = _deps(FakeOcr(), agent, verifier)
    async with factory() as session:
        outcome = await process_page(
            session,
            job_id=job_id,
            page_number=1,
            pdf_path=pdf,
            renders_dir=tmp_path / "r",
            deps=deps,
        )
    assert outcome.status is PageStatus.VERIFIED and outcome.retries_used == 0
    assert len(agent.calls) == 2
    assert "CORRECTIVE" in agent.calls[1]["context"]
    assert outcome.markdown == "# P1 fixed"


# --- 4.2a: deterministic coverage floor ---


async def test_floor_overrules_judge_pass(db, tmp_path: Path) -> None:
    """Judge says 99/pass, but the markdown omits half the OCR text:
    the floor must force a retry and finally needs_review."""
    factory, created = db
    pdf = _pdf(tmp_path, pages=1)
    job_id = await _job(factory, created, tmp_path, pages=1)
    ocr_text = " ".join(f"token{i}" for i in range(40))  # 40 measurable tokens
    half = " ".join(f"token{i}" for i in range(20))  # 50% recall
    agent = FakeAgent({1: [_env(f"# half\n{half}")] * 3})
    verifier = FakeAgent({1: [_verdict(99)] * 3})
    deps = _deps(
        FakeOcr(text=ocr_text),
        agent,
        verifier,
        max_page_retries=2,
        coverage_floor=0.80,
        coverage_floor_min_ocr_tokens=30,
    )
    async with factory() as session:
        outcome = await process_page(
            session,
            job_id=job_id,
            page_number=1,
            pdf_path=pdf,
            renders_dir=tmp_path / "r",
            deps=deps,
        )
    assert outcome.status is PageStatus.NEEDS_REVIEW
    assert len(agent.calls) == 3  # full retry cycle, cap honored
    # retry context names the floor failure
    assert "coverage floor" in agent.calls[1]["context"]
    async with factory() as session:
        row = await session.scalar(select(Page).where(Page.job_id == job_id))
        assert row is not None and row.needs_review is True
        assert row.omissions["floor_score"] is not None and row.omissions["floor_score"] < 0.8


async def test_floor_recovery_on_retry(db, tmp_path: Path) -> None:
    """Floor fails on attempt 1, agent re-emits full text, page verifies."""
    factory, created = db
    pdf = _pdf(tmp_path, pages=1)
    job_id = await _job(factory, created, tmp_path, pages=1)
    ocr_text = " ".join(f"token{i}" for i in range(40))
    half = " ".join(f"token{i}" for i in range(20))
    agent = FakeAgent({1: [_env(f"# half\n{half}"), _env(f"# full\n{ocr_text}")]})
    verifier = FakeAgent({1: [_verdict(99)]})
    deps = _deps(
        FakeOcr(text=ocr_text),
        agent,
        verifier,
        coverage_floor=0.80,
        coverage_floor_min_ocr_tokens=30,
    )
    async with factory() as session:
        outcome = await process_page(
            session,
            job_id=job_id,
            page_number=1,
            pdf_path=pdf,
            renders_dir=tmp_path / "r",
            deps=deps,
        )
    assert outcome.status is PageStatus.VERIFIED and outcome.retries_used == 1
    assert "coverage floor" in agent.calls[1]["context"]


async def test_floor_unmeasurable_never_gates(db, tmp_path: Path) -> None:
    """Short OCR reference (under min tokens): floor returns None and a
    judge-pass alone verifies (existing behavior preserved)."""
    factory, created = db
    pdf = _pdf(tmp_path, pages=1)
    job_id = await _job(factory, created, tmp_path, pages=1)
    agent = FakeAgent({1: [_env("# P1\nbody")]})
    verifier = FakeAgent({1: [_verdict(97)]})
    deps = _deps(
        FakeOcr(text="tiny"),  # 1 token — unmeasurable
        agent,
        verifier,
        coverage_floor=0.80,
    )
    async with factory() as session:
        outcome = await process_page(
            session,
            job_id=job_id,
            page_number=1,
            pdf_path=pdf,
            renders_dir=tmp_path / "r",
            deps=deps,
        )
    assert outcome.status is PageStatus.VERIFIED and outcome.retries_used == 0


async def test_floor_disabled_never_gates(db, tmp_path: Path) -> None:
    """coverage_floor=0: even 1% recall must not block a judge-pass."""
    factory, created = db
    pdf = _pdf(tmp_path, pages=1)
    job_id = await _job(factory, created, tmp_path, pages=1)
    ocr_text = " ".join(f"token{i}" for i in range(40))
    agent = FakeAgent({1: [_env("# one word")]})
    verifier = FakeAgent({1: [_verdict(99)]})
    deps = _deps(
        FakeOcr(text=ocr_text),
        agent,
        verifier,
        coverage_floor=0.0,
        coverage_floor_min_ocr_tokens=30,
    )
    async with factory() as session:
        outcome = await process_page(
            session,
            job_id=job_id,
            page_number=1,
            pdf_path=pdf,
            renders_dir=tmp_path / "r",
            deps=deps,
        )
    assert outcome.status is PageStatus.VERIFIED and outcome.retries_used == 0


# --- 4.3 / 4.5 ---


def _control() -> JobControl:
    return JobControl(JobStatus.OCR)


async def test_run_pages_cancel_mid_page_resumes(db, tmp_path: Path) -> None:
    factory, created = db
    pdf = _pdf(tmp_path, pages=4)
    job_id = await _job(factory, created, tmp_path, pages=4)
    control = _control()

    async def _hook(page_number, call):
        if page_number == 2:
            control.request_cancel()

    raws = {n: [_env(f"# P{n}\nbody{n}")] for n in (1, 2, 3, 4)}
    verdicts = {n: [_verdict(97)] for n in (1, 2, 3, 4)}
    deps = _deps(FakeOcr(), FakeAgent(raws, hook=_hook), FakeAgent(verdicts))
    summary = await run_pages(
        factory,
        job_id=job_id,
        pdf_path=pdf,
        renders_dir=tmp_path / "r",
        deps=deps,
        control=control,
    )
    assert summary.stopped == "cancelled" and summary.done_pages == [1, 2]

    control2 = _control()
    deps2 = _deps(FakeOcr(), FakeAgent(raws), FakeAgent(verdicts))
    summary2 = await run_pages(
        factory,
        job_id=job_id,
        pdf_path=pdf,
        renders_dir=tmp_path / "r",
        deps=deps2,
        control=control2,
    )
    assert summary2.stopped == "completed" and summary2.done_pages == [3, 4]
    async with factory() as session:
        rows = (
            await session.scalars(
                select(Page).where(Page.job_id == job_id).order_by(Page.page_number)
            )
        ).all()
        assert [r.markdown for r in rows] == [f"# P{n}\nbody{n}" for n in (1, 2, 3, 4)]


async def test_run_pages_pause_then_resume(db, tmp_path: Path) -> None:
    factory, created = db
    pdf = _pdf(tmp_path, pages=2)
    job_id = await _job(factory, created, tmp_path, pages=2)
    control = _control()
    control.request_pause()
    raws = {n: [_env(f"# P{n}")] for n in (1, 2)}
    deps = _deps(FakeOcr(), FakeAgent(raws), FakeAgent({n: [_verdict(99)] for n in (1, 2)}))
    task = asyncio.ensure_future(
        run_pages(
            factory,
            job_id=job_id,
            pdf_path=pdf,
            renders_dir=tmp_path / "r",
            deps=deps,
            control=control,
        )
    )
    await asyncio.sleep(0.1)
    assert not task.done()
    control.request_resume(JobStatus.TRANSCRIBING)
    summary = await asyncio.wait_for(task, timeout=30.0)
    assert summary.done_pages == [1, 2]


async def test_run_pages_rolling_context_and_outline(db, tmp_path: Path) -> None:
    factory, created = db
    pdf = _pdf(tmp_path, pages=3)
    job_id = await _job(factory, created, tmp_path, pages=3)
    agent = FakeAgent({n: [_env(f"# P{n}\nbody")] for n in (1, 2, 3)})
    deps = _deps(
        FakeOcr(), agent, FakeAgent({n: [_verdict(99)] for n in (1, 2, 3)}), rolling_context_pages=1
    )
    await run_pages(
        factory,
        job_id=job_id,
        pdf_path=pdf,
        renders_dir=tmp_path / "r",
        deps=deps,
        control=_control(),
    )
    third = next(call for call in agent.calls if call["page"] == 3)
    assert "# P2" in third["context"] and "# P1" not in third["context"]
    assert "# P1" in third["outline"] and "# P2" in third["outline"]


async def test_on_page_start_fires_in_order(db, tmp_path: Path) -> None:
    factory, created = db
    pdf = _pdf(tmp_path, pages=2)
    job_id = await _job(factory, created, tmp_path, pages=2)
    started: list[int] = []
    raws = {n: [_env(f"# P{n}")] for n in (1, 2)}
    deps = _deps(FakeOcr(), FakeAgent(raws), FakeAgent({n: [_verdict(99)] for n in (1, 2)}))

    async def _started(page_number: int) -> None:
        started.append(page_number)

    summary = await run_pages(
        factory,
        job_id=job_id,
        pdf_path=pdf,
        renders_dir=tmp_path / "r",
        deps=deps,
        control=_control(),
        on_page_start=_started,
    )
    assert summary.done_pages == [1, 2]
    assert started == [1, 2]


async def test_sustained_outage_pauses_job(db, tmp_path: Path) -> None:
    factory, created = db
    pdf = _pdf(tmp_path, pages=2)
    job_id = await _job(factory, created, tmp_path, pages=2)
    deps = _deps(FakeOcr(fail=httpx.ConnectError("down")), FakeAgent({}), FakeAgent({}))
    summary = await run_pages(
        factory,
        job_id=job_id,
        pdf_path=pdf,
        renders_dir=tmp_path / "r",
        deps=deps,
        control=_control(),
    )
    assert summary.stopped == "paused" and summary.done_pages == []
    async with factory() as session:
        assert await repo.find_resume_point(session, job_id) == 1
        job = await repo.get_job(session, job_id)
        assert job is not None and job.status is JobStatus.PAUSED


# --- 4.6: hybrid routing ---


async def test_hybrid_born_digital_page_verifies_deterministically(db, tmp_path: Path) -> None:
    """Rich native text + verifier pass → page verifies with ZERO agent
    and ZERO OCR calls (the entire cost saving of routing)."""
    factory, created = db
    pdf = _rich_pdf(tmp_path, pages=1)
    job_id = await _job(factory, created, tmp_path, pages=1)
    verifier = FakeAgent({1: [_verdict(99)]})

    async def _boom_ocr(*a, **k):
        raise AssertionError("OCR must never run on a routed deterministic page")

    async def _boom_agent(*a, **k):
        raise AssertionError("transcribe must never run on a routed deterministic page")

    deps = _deps(_boom_ocr, _boom_agent, verifier, hybrid_routing=True)
    async with factory() as session:
        outcome = await process_page(
            session,
            job_id=job_id,
            page_number=1,
            pdf_path=pdf,
            renders_dir=tmp_path / "r",
            deps=deps,
        )
    assert outcome.status is PageStatus.VERIFIED and outcome.route == "deterministic"
    assert outcome.retries_used == 0
    assert len(verifier.calls) == 1  # exactly one verify (the gate)
    async with factory() as session:
        row = await session.scalar(select(Page).where(Page.job_id == job_id))
        assert row is not None and row.status is PageStatus.VERIFIED
        assert row.omissions["route"] == "deterministic"
        assert row.omissions["floor_score"] is not None and row.omissions["floor_score"] >= 0.8
        assert row.timings["model"] == "deterministic"


async def test_hybrid_verifier_rejection_escalates_to_agentic(db, tmp_path: Path) -> None:
    """Verifier rejects the deterministic candidate → full agentic cycle
    runs (OCR + transcribe), verifier misses carried as a head start."""
    factory, created = db
    pdf = _rich_pdf(tmp_path, pages=1)
    job_id = await _job(factory, created, tmp_path, pages=1)
    ocr_text = " ".join(f"rich00word{i:03d}" for i in range(40))
    agent = FakeAgent({1: [_env("# Agentic\n" + ocr_text)]})
    verifier = FakeAgent({1: [_verdict(60, "retry", ["flat table"]), _verdict(98)]})
    deps = _deps(
        FakeOcr(text=ocr_text),
        agent,
        verifier,
        hybrid_routing=True,
        max_page_retries=1,
    )
    async with factory() as session:
        outcome = await process_page(
            session,
            job_id=job_id,
            page_number=1,
            pdf_path=pdf,
            renders_dir=tmp_path / "r",
            deps=deps,
        )
    assert outcome.status is PageStatus.VERIFIED and outcome.route == "agentic"
    assert outcome.markdown.startswith("# Agentic")
    # escalation note reached the agent context
    assert "deterministic extraction" in agent.calls[0]["context"]
    # two verifier calls: deterministic gate + agentic verify
    assert len(verifier.calls) == 2


async def test_hybrid_scanned_page_never_tries_deterministic(db, tmp_path: Path) -> None:
    """Sparse native text → decide_route says no → straight to OCR+agent."""
    factory, created = db
    pdf = _pdf(tmp_path, pages=1)  # insert_text page: ~10 words
    job_id = await _job(factory, created, tmp_path, pages=1)
    body = " ".join(f"w{i:03d}" for i in range(40))
    agent = FakeAgent({1: [_env(f"# P1\n{body}")]})
    verifier = FakeAgent({1: [_verdict(97)]})
    deps = _deps(FakeOcr(text=body), agent, verifier)
    deps.hybrid_routing = True
    async with factory() as session:
        outcome = await process_page(
            session,
            job_id=job_id,
            page_number=1,
            pdf_path=pdf,
            renders_dir=tmp_path / "r",
            deps=deps,
        )
    assert outcome.status is PageStatus.VERIFIED and outcome.route == "agentic"


async def test_hybrid_disabled_preserves_original_path(db, tmp_path: Path) -> None:
    """hybrid_routing=False (default): even a rich born-digital page runs
    the full agentic path — behavior identical to pre-routing code."""
    factory, created = db
    pdf = _rich_pdf(tmp_path, pages=1)
    job_id = await _job(factory, created, tmp_path, pages=1)
    agent = FakeAgent({1: [_env("# P1\nbody")]})
    verifier = FakeAgent({1: [_verdict(97)]})
    deps = _deps(FakeOcr(), agent, verifier)  # hybrid_routing defaults False
    async with factory() as session:
        outcome = await process_page(
            session,
            job_id=job_id,
            page_number=1,
            pdf_path=pdf,
            renders_dir=tmp_path / "r",
            deps=deps,
        )
    assert outcome.status is PageStatus.VERIFIED and outcome.route == "agentic"
    assert len(agent.calls) == 1  # transcribe ran


# --- 4.4 ---


async def test_lookahead_equals_sequential(db, tmp_path: Path) -> None:
    factory, _created = db
    pdf = _pdf(tmp_path, pages=3)
    raws = {n: [_env(f"# P{n}\nbody{n}")] for n in (1, 2, 3)}
    verdicts = {n: [_verdict(96)] for n in (1, 2, 3)}

    async def _run(job_tag, runner, offset):
        async with factory() as session:
            job = await repo.create_job(
                session,
                filename=f"m4-look-{job_tag}-{uuid.uuid4().hex}.pdf",
                file_sha256="g" * 64,
                page_count=3,
                output_dir=str(tmp_path),
            )
            jid = job.id
        deps = _deps(FakeOcr(), FakeAgent(raws), FakeAgent(verdicts))
        summary = await runner(
            factory,
            job_id=jid,
            pdf_path=pdf,
            renders_dir=tmp_path / f"r{offset}",
            deps=deps,
            control=_control(),
        )
        async with factory() as session:
            rows = (
                await session.scalars(
                    select(Page).where(Page.job_id == jid).order_by(Page.page_number)
                )
            ).all()
            texts = [r.markdown for r in rows]
        async with factory() as session:
            doomed = await session.get(Job, jid)
            if doomed is not None:
                await session.delete(doomed)
            await session.commit()
        return summary, texts

    seq_summary, seq_texts = await _run("seq", run_pages, "s")
    look_summary, look_texts = await _run("look", run_pages_lookahead, "l")
    assert seq_summary.done_pages == look_summary.done_pages == [1, 2, 3]
    assert seq_texts == look_texts == [f"# P{n}\nbody{n}" for n in (1, 2, 3)]


async def test_lookahead_overlaps_ocr_and_transcribe(db, tmp_path: Path) -> None:
    factory, created = db
    pdf = _pdf(tmp_path, pages=2)
    job_id = await _job(factory, created, tmp_path, pages=2)
    ocr2_started = asyncio.Event()

    class OverlapOcr(FakeOcr):
        async def run_page(self, image, prompt="Text Recognition:"):
            if "page-002" in str(image):
                ocr2_started.set()
            return await super().run_page(image, prompt)

    # Deterministic gate: page-1 transcribe waits for ocr-2 start.
    async def _hook2(page_number, call):
        if page_number == 1:
            await asyncio.wait_for(ocr2_started.wait(), timeout=10.0)

    deps = _deps(
        OverlapOcr(),
        FakeAgent({1: [_env("# P1")], 2: [_env("# P2")]}, hook=_hook2),
        FakeAgent({1: [_verdict(99)], 2: [_verdict(99)]}),
    )
    summary = await run_pages_lookahead(
        factory,
        job_id=job_id,
        pdf_path=pdf,
        renders_dir=tmp_path / "r",
        deps=deps,
        control=_control(),
    )
    # No timeout fired above: OCR(2) provably started while transcribe(1) ran.
    assert summary.done_pages == [1, 2]
