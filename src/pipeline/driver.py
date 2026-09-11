"""Full-chain job driver — preflight to deliverable with progress.

Composes the pipeline stages for one job: preflight → per-page loop (real
adapters) → image resolution → dedup/assemble → Cloud QA → normalize →
lint/integrity gate → report → deliverable → embeddings. Emits §5.2 events
via the injected emit callable; honors pause/cancel between stages (inside
pages via JobControl). Any PauseJob from the page loop is already persisted
there; unexpected exceptions fail the job with the error recorded (never a
bare traceback to the client).
"""

import datetime
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.chroma import ChromaStore
from src.config import Settings
from src.db import repo
from src.db.models import JobStatus
from src.logging import get_logger, set_context
from src.pdf import preflight
from src.pipeline.agent import GlmFlashAgent
from src.pipeline.assemble import assemble, dedup_furniture, make_chroma_similar
from src.pipeline.control import IllegalTransitionError, JobControl
from src.pipeline.gfm import classify, lint_markdown, normalize_markdown
from src.pipeline.images import FigurePayload, check_assets, resolve_page_figures
from src.pipeline.ocr import GlmOcrEngine
from src.pipeline.pages import PageDeps, PageOutcome, RunSummary, run_pages
from src.pipeline.prompts import (
    PROMPT_VERSIONS,
    QA_TEMPLATE,
    VERIFICATION_TEMPLATE,
    compute_pipeline_version,
)
from src.pipeline.qa import run_qa_pass
from src.pipeline.report import (
    build_report,
    collect_job_pages,
    persist_document_embeddings,
    write_output,
)
from src.workspace import Workspace

Effort = Literal["low", "medium", "high"]

Emit = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True)
class JobOptions:
    """Per-job knobs (upload form overrides Settings defaults)."""

    render_dpi: int = 200
    coverage_threshold: int = 95
    coverage_floor: float = 0.80
    max_page_retries: int = 2
    rolling_context_pages: int = 3
    toc_enabled: bool = True
    fig_details: bool = True
    hybrid_routing: bool = False
    thinking_transcribe: Effort = "low"
    thinking_qa: Effort = "high"

    @staticmethod
    def from_settings(settings: Settings, overrides: dict[str, Any] | None = None) -> JobOptions:
        """Defaults from Settings, sparse validated overrides on top."""
        base: dict[str, Any] = {
            "render_dpi": settings.RENDER_DPI,
            "coverage_threshold": settings.COVERAGE_THRESHOLD,
            "coverage_floor": settings.COVERAGE_FLOOR_TOKENS / 100.0,
            "max_page_retries": settings.MAX_PAGE_RETRIES,
            "rolling_context_pages": settings.ROLLING_CONTEXT_PAGES,
            "toc_enabled": settings.TOC_ENABLED,
            "fig_details": settings.FIG_DETAILS_BLOCKS,
            "hybrid_routing": settings.HYBRID_ROUTING,
            "thinking_transcribe": settings.THINKING_EFFORT_TRANSCRIBE,
            "thinking_qa": settings.THINKING_EFFORT_QA,
        }
        for key, value in (overrides or {}).items():
            if key in base and type(value) is type(base[key]):
                base[key] = value
        return JobOptions(**base)


async def _goto(
    factory: async_sessionmaker[AsyncSession],
    control: JobControl,
    job_id: UUID,
    target: JobStatus,
    emit: Emit,
    stage: str,
) -> None:
    async with factory() as session:
        await repo.set_job_status(session, job_id, target, stage)
        await repo.log_event(session, job_id, "info", f"stage: {stage}", stage=stage)
    set_context(stage=stage)
    get_logger("driver").info("stage: %s", stage)
    try:
        control.transition(target)
    except IllegalTransitionError:
        pass  # DB status is authoritative; control follows when on-map
    await emit({"event": "stage_changed", "job_id": str(job_id), "stage": stage})


async def run_job(
    *,
    factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    workspace: Workspace,
    job_id: UUID,
    pdf_path: Path,
    output_dir: Path,
    options: JobOptions,
    control: JobControl,
    emit: Emit,
) -> str:
    """Run the whole chain; returns the final job status value."""
    total = 0
    logger = get_logger("driver")
    set_context(job_id=str(job_id))
    try:
        async with factory() as session:
            existing = await repo.get_job(session, job_id)
            if existing is not None and existing.status is JobStatus.COMPLETED:
                logger.info("job already completed; nothing to do")
                await emit({"event": "job_finished", "job_id": str(job_id), "status": "completed"})
                return "completed"
        await control.wait_if_paused()  # paused while queued: hold before preflight
        if control.cancel_requested:
            return await _cancelled(factory, control, job_id, emit)
        await _goto(factory, control, job_id, JobStatus.PREFLIGHT, emit, "preflight")
        pre = preflight(pdf_path)
        if not pre.valid:
            logger.warning("preflight failed: %s", pre.error)
            return await _fail(factory, control, job_id, emit, f"preflight: {pre.error}")
        total = pre.page_count
        logger.info("job started: %s pages=%d", pdf_path.name, total)
        async with factory() as session:
            job = await repo.get_job(session, job_id)
            if job is not None and job.page_count != total:
                job.page_count = total
                await session.commit()

        job_dir = workspace.job_dir(str(job_id))
        renders_dir = job_dir / "renders"
        assets_staging = job_dir / "assets"

        await _goto(factory, control, job_id, JobStatus.RENDERING, emit, "rendering")
        await _goto(factory, control, job_id, JobStatus.OCR, emit, "ocr")
        key = settings.OLLAMA_API_KEY.get_secret_value()
        deps = PageDeps(
            ocr=GlmOcrEngine(
                settings.OLLAMA_LOCAL_URL,
                model=settings.OCR_MODEL,
                timeout=settings.OCR_TIMEOUT_SECONDS,
            ),
            agent=GlmFlashAgent(
                settings.OLLAMA_CLOUD_URL,
                key,
                model=settings.AGENT_MODEL,
                thinking_effort=options.thinking_transcribe,
                timeout=settings.AGENT_TIMEOUT_SECONDS,
            ),
            verifier=GlmFlashAgent(
                settings.OLLAMA_CLOUD_URL,
                key,
                model=settings.AGENT_MODEL,
                thinking_effort=options.thinking_transcribe,
                timeout=settings.AGENT_TIMEOUT_SECONDS,
                system_prompt=VERIFICATION_TEMPLATE,
            ),
            render_dpi=options.render_dpi,
            coverage_threshold=options.coverage_threshold,
            coverage_floor=options.coverage_floor,
            max_page_retries=options.max_page_retries,
            rolling_context_pages=options.rolling_context_pages,
            hybrid_routing=options.hybrid_routing,
        )

        async def _on_page(outcome: PageOutcome) -> None:
            logger.info(
                "page %d/%d %s coverage=%s retries=%d",
                outcome.page_number,
                total,
                outcome.status.value,
                outcome.coverage,
                outcome.retries_used,
            )
            await emit(
                {
                    "event": "page_update",
                    "job_id": str(job_id),
                    "page": outcome.page_number,
                    "total": total,
                    "status": outcome.status.value,
                    "coverage": outcome.coverage,
                }
            )
            if outcome.markdown and not outcome.skipped:
                await emit(
                    {
                        "event": "page_markdown",
                        "job_id": str(job_id),
                        "page": outcome.page_number,
                        "markdown": outcome.markdown,
                    }
                )

        async def _on_page_start(page_number: int) -> None:
            set_context(page=page_number)
            logger.info(
                "transcribing page %d/%d (render → OCR → agent → verify)", page_number, total
            )
            await emit(
                {
                    "event": "page_update",
                    "job_id": str(job_id),
                    "page": page_number,
                    "total": total,
                    "status": "processing",
                    "coverage": None,
                }
            )
            await emit(
                {
                    "event": "log",
                    "job_id": str(job_id),
                    "level": "info",
                    "message": f"transcribing page {page_number}/{total}",
                }
            )

        summary: RunSummary = await run_pages(
            factory,
            job_id=job_id,
            pdf_path=pdf_path,
            renders_dir=renders_dir,
            deps=deps,
            control=control,
            on_page=_on_page,
            on_page_start=_on_page_start,
        )
        if summary.stopped != "completed":
            await emit({"event": "job_finished", "job_id": str(job_id), "status": summary.stopped})
            return summary.stopped

        await _goto(factory, control, job_id, JobStatus.ASSEMBLING, emit, "assembling")
        await control.wait_if_paused()
        if control.cancel_requested:
            return await _cancelled(factory, control, job_id, emit)
        async with factory() as session:
            job_row, pages = await collect_job_pages(session, job_id)
            assert job_row is not None
            await repo.clear_job_images(session, job_id)  # idempotent re-entry (8.2)
            resolved: list[str] = []
            orphans: list[dict[str, object]] = []
            for page in sorted(pages, key=lambda p: p.page_number):
                figs = _payloads(page.omissions)
                outcome = await resolve_page_figures(
                    session,
                    job_id=job_id,
                    page_number=page.page_number,
                    markdown=page.markdown or "",
                    figures=figs,
                    pdf_path=pdf_path,
                    renders_dir=renders_dir,
                    assets_dir=assets_staging,
                    fig_details=options.fig_details,
                )
                resolved.append(outcome.markdown)
                for entry in outcome.orphaned:
                    note = dict(entry)
                    note["page"] = page.page_number
                    orphans.append(note)
                    await repo.log_event(
                        session,
                        job_id,
                        "warning",
                        f"page {page.page_number}: figure without placeholder appended "
                        f"at page end ({note.get('reason')})",
                        stage="assembling",
                    )
            await session.commit()

        store = ChromaStore(
            str(settings.CHROMA_PATH), settings.EMBED_MODEL, settings.OLLAMA_LOCAL_URL
        )
        dedup = dedup_furniture(resolved, make_chroma_similar(store))
        assembled = assemble(dedup.pages, toc_enabled=options.toc_enabled)

        await _goto(factory, control, job_id, JobStatus.QA, emit, "qa")
        await control.wait_if_paused()
        if control.cancel_requested:
            return await _cancelled(factory, control, job_id, emit)
        qa_agent = GlmFlashAgent(
            settings.OLLAMA_CLOUD_URL,
            key,
            model=settings.AGENT_MODEL,
            thinking_effort=options.thinking_qa,
            timeout=settings.AGENT_TIMEOUT_SECONDS,
            system_prompt=QA_TEMPLATE,
        )
        outline = "\n".join(h for text in dedup.pages for h in _headings(text))
        qa = await run_qa_pass(qa_agent, assembled, outline)
        normalized = normalize_markdown(qa.markdown)
        warnings = lint_markdown(normalized)
        assets_staging.mkdir(parents=True, exist_ok=True)
        integrity = check_assets(normalized, assets_staging)
        verdict = classify(
            integrity.ok,
            integrity.missing_files + integrity.orphan_files + integrity.leftover_placeholders,
            warnings,
        )
        async with factory() as session:
            job_row, pages = await collect_job_pages(session, job_id)
            assert job_row is not None
            pipeline_version = compute_pipeline_version(
                agent_model=settings.AGENT_MODEL,
                ocr_model=settings.OCR_MODEL,
                embed_model=settings.EMBED_MODEL,
                render_dpi=options.render_dpi,
                rolling_context_pages=options.rolling_context_pages,
                coverage_threshold=options.coverage_threshold,
                coverage_floor=options.coverage_floor,
                hybrid_routing=options.hybrid_routing,
                max_page_retries=options.max_page_retries,
                thinking_transcribe=options.thinking_transcribe,
                thinking_qa=options.thinking_qa,
                toc_enabled=options.toc_enabled,
                fig_details=options.fig_details,
            )
            report = build_report(
                job_row,
                pages,
                pipeline_version=pipeline_version,
                prompt_versions=dict(PROMPT_VERSIONS),
                furniture_removed=[dict(e) for e in dedup.removed],
                qa_applied=len(qa.applied),
                qa_rejected=len(qa.rejected),
                qa_log=[
                    {
                        "section": p.section,
                        "find": p.find,
                        "replace": p.replace,
                        "applied": p.applied,
                        "reason": p.reason,
                    }
                    for p in (*qa.applied, *qa.rejected)
                ],
                lint_warnings=warnings,
                orphaned_figures=orphans,
            )
            if verdict.hard_fail:
                job_row.status = JobStatus.FAILED
                job_row.error = {"qa": verdict.reasons}
                await session.commit()
                await emit({"event": "job_finished", "job_id": str(job_id), "status": "failed"})
                return "failed"
            try:
                deliverable = write_output(
                    output_dir, pdf_path.stem, normalized, assets_staging, report
                )
            except OSError as exc:
                # Output unwritable / disk full (§8): pause, don't fail —
                # resume after the operator fixes the disk.
                async with factory() as session:
                    await repo.set_job_status(session, job_id, JobStatus.PAUSED, "qa")
                try:
                    control.request_pause()
                except IllegalTransitionError:
                    pass
                await emit(
                    {
                        "event": "job_finished",
                        "job_id": str(job_id),
                        "status": "paused",
                        "error": f"output unwritable: {exc}",
                    }
                )
                return "paused"
            try:
                persist_document_embeddings(store, pdf_path.stem, normalized)
            except Exception as exc:  # noqa: BLE001 - optional post-step, job continues
                await repo.log_event(
                    session, job_id, "warning", f"doc embeddings skipped: {exc}", stage="qa"
                )
            options_snapshot = dict(job_row.options or {})
            options_snapshot["results"] = {
                "furniture_removed": [dict(e) for e in dedup.removed],
                "qa_applied": len(qa.applied),
                "qa_rejected": len(qa.rejected),
                "qa_log": [
                    {
                        "section": p.section,
                        "find": p.find,
                        "replace": p.replace,
                        "applied": p.applied,
                        "reason": p.reason,
                    }
                    for p in (*qa.applied, *qa.rejected)
                ],
                "lint_warnings": warnings,
                "orphaned_figures": orphans,
                "artifacts": {
                    "document": str(deliverable.markdown_path),
                    "report": str(deliverable.report_path),
                },
            }
            job_row.options = options_snapshot
            job_row.status = JobStatus.COMPLETED
            job_row.finished_at = datetime.datetime.now(datetime.UTC)
            await session.commit()
        workspace.cleanup(str(job_id), keep=settings.KEEP_WORKSPACE_ON_SUCCESS)
        get_logger("driver").info("completed: %s", deliverable.markdown_path)
        await emit(
            {
                "event": "job_finished",
                "job_id": str(job_id),
                "status": "completed",
                "output_path": str(deliverable.markdown_path),
            }
        )
        await _goto(factory, control, job_id, JobStatus.COMPLETED, emit, "completed")
        return "completed"
    except Exception as exc:
        # Diagnostic: log full traceback so the next ReadError pinpoints
        # the originating call (ocr vs transcribe vs verify). Behavior
        # unchanged — still fails the job via _fail.
        logger.warning("unexpected failure: %s", exc, exc_info=True)
        return await _fail(factory, control, job_id, emit, f"{type(exc).__name__}: {exc}")


def _payloads(omissions: object) -> list[FigurePayload]:
    if not isinstance(omissions, dict):
        return []
    figs = omissions.get("figures", [])
    payloads = []
    for fig in figs if isinstance(figs, list) else []:
        try:
            bbox = tuple(float(v) for v in fig["bbox"])
            if len(bbox) != 4:
                continue
            payloads.append(
                FigurePayload(
                    index=int(fig["index"]),
                    bbox=bbox,
                    alt=str(fig.get("alt", "")),
                    caption=fig.get("caption"),
                )
            )
        except KeyError, TypeError, ValueError:
            continue
    return payloads


def _headings(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("#")]


async def _fail(
    factory: async_sessionmaker[AsyncSession],
    control: JobControl,
    job_id: UUID,
    emit: Emit,
    detail: str,
) -> str:
    get_logger("driver").warning("failed: %s", detail)
    async with factory() as session:
        job = await repo.get_job(session, job_id)
        if job is not None:
            job.status = JobStatus.FAILED
            job.error = {"error": detail}
            await session.commit()
    try:
        control.transition(JobStatus.FAILED)
    except IllegalTransitionError:
        pass
    await emit(
        {"event": "job_finished", "job_id": str(job_id), "status": "failed", "error": detail}
    )
    return "failed"


async def _cancelled(
    factory: async_sessionmaker[AsyncSession], control: JobControl, job_id: UUID, emit: Emit
) -> str:
    get_logger("driver").info("cancelled")
    async with factory() as session:
        await repo.set_job_status(session, job_id, JobStatus.CANCELLED, "cancelled")
    try:
        control.request_cancel()
    except IllegalTransitionError:
        pass
    await emit({"event": "job_finished", "job_id": str(job_id), "status": "cancelled"})
    return "cancelled"


__all__ = ["Effort", "JobOptions", "run_job"]
