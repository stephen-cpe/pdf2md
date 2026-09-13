"""Full-chain job driver — preflight to deliverable with progress.

Composes the pipeline stages for one job: preflight → per-page loop (real
adapters) → per-figure diagram→Mermaid reinterpretation → image resolution →
dedup/assemble → normalize → lint/integrity gate → report → deliverable.
Emits §5.2 events via the injected emit callable; honors pause/cancel between
stages (inside pages via JobControl). Any PauseJob from the page loop is
already persisted there; unexpected exceptions fail the job with the error
recorded (never a bare traceback to the client).
"""

import datetime
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import Settings
from src.db import repo
from src.db.models import JobStatus
from src.logging import get_logger, set_context
from src.pdf import preflight
from src.pipeline.agent import GlmFlashAgent
from src.pipeline.assemble import assemble, dedup_furniture, make_exact_similar
from src.pipeline.control import IllegalTransitionError, JobControl
from src.pipeline.diagrams import DiagramResult, convert_figure
from src.pipeline.gfm import classify, lint_markdown, normalize_markdown
from src.pipeline.images import FigurePayload, check_assets, resolve_page_figures
from src.pipeline.ocr import GlmOcrEngine
from src.pipeline.pages import PageDeps, PageOutcome, RunSummary, run_pages
from src.pipeline.prompts import (
    DIAGRAM_TEMPLATE,
    DIAGRAM_VERIFY_TEMPLATE,
    PROMPT_VERSIONS,
    VERIFICATION_TEMPLATE,
    compute_pipeline_version,
)
from src.pipeline.report import (
    build_report,
    collect_job_images,
    collect_job_pages,
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
    thinking_transcribe: Effort = "low"
    thinking_diagram: Effort = "high"
    native_text_first: bool = True
    native_text_min_words: int = 20
    ocr_enabled: bool = True
    diagram_to_mermaid: bool = True
    diagram_min_confidence: int = 80
    diagram_verify: bool = True
    diagram_fallback: str = "both"
    diagram_keep_image: bool = True

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
            "thinking_transcribe": settings.THINKING_EFFORT_TRANSCRIBE,
            "thinking_diagram": settings.THINKING_EFFORT_DIAGRAM,
            "native_text_first": settings.NATIVE_TEXT_FIRST,
            "native_text_min_words": settings.NATIVE_TEXT_MIN_WORDS,
            "ocr_enabled": settings.OCR_ENABLED,
            "diagram_to_mermaid": settings.DIAGRAM_TO_MERMAID,
            "diagram_min_confidence": settings.DIAGRAM_MIN_CONFIDENCE,
            "diagram_verify": settings.DIAGRAM_VERIFY,
            "diagram_fallback": settings.DIAGRAM_FALLBACK,
            "diagram_keep_image": settings.DIAGRAM_KEEP_IMAGE,
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
            native_text_first=options.native_text_first,
            native_text_min_words=options.native_text_min_words,
            ocr_enabled=options.ocr_enabled,
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
        key = settings.OLLAMA_API_KEY.get_secret_value()
        async with factory() as session:
            job_row, pages = await collect_job_pages(session, job_id)
            assert job_row is not None
            await repo.clear_job_images(session, job_id)  # idempotent re-entry (8.2)
            resolved: list[str] = []
            orphans: list[dict[str, object]] = []
            for page in sorted(pages, key=lambda p: p.page_number):
                figs = await _diagram_payloads(
                    settings=settings,
                    key=key,
                    job_id=job_id,
                    job_dir=job_dir,
                    pdf_path=pdf_path,
                    renders_dir=renders_dir,
                    page_number=page.page_number,
                    omissions=page.omissions,
                    options=options,
                    emit=emit,
                    total=total,
                )
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
                    fallback=options.diagram_fallback,
                    keep_image=options.diagram_keep_image,
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

        dedup = dedup_furniture(resolved, make_exact_similar())
        assembled = assemble(dedup.pages, toc_enabled=options.toc_enabled)
        normalized = normalize_markdown(assembled)
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
            images = await collect_job_images(session, job_id)
            pipeline_version = compute_pipeline_version(
                agent_model=settings.AGENT_MODEL,
                ocr_model=settings.OCR_MODEL,
                render_dpi=options.render_dpi,
                rolling_context_pages=options.rolling_context_pages,
                coverage_threshold=options.coverage_threshold,
                coverage_floor=options.coverage_floor,
                max_page_retries=options.max_page_retries,
                thinking_transcribe=options.thinking_transcribe,
                thinking_diagram=options.thinking_diagram,
                native_text_first=options.native_text_first,
                native_text_min_words=options.native_text_min_words,
                ocr_enabled=options.ocr_enabled,
                toc_enabled=options.toc_enabled,
                fig_details=options.fig_details,
                diagram_to_mermaid=options.diagram_to_mermaid,
                diagram_min_confidence=options.diagram_min_confidence,
                diagram_verify=options.diagram_verify,
                diagram_fallback=options.diagram_fallback,
                diagram_keep_image=options.diagram_keep_image,
            )
            report = build_report(
                job_row,
                pages,
                pipeline_version=pipeline_version,
                prompt_versions=dict(PROMPT_VERSIONS),
                furniture_removed=[dict(e) for e in dedup.removed],
                lint_warnings=warnings,
                images=images,
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
                    await repo.set_job_status(session, job_id, JobStatus.PAUSED, "assembling")
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
            mermaid_count = sum(1 for img in images if img.conversion_status == "mermaid")
            options_snapshot = dict(job_row.options or {})
            options_snapshot["results"] = {
                "furniture_removed": [dict(e) for e in dedup.removed],
                "lint_warnings": warnings,
                "orphaned_figures": orphans,
                "figures_total": len(images),
                "figures_mermaid": mermaid_count,
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


async def _diagram_payloads(
    *,
    settings: Settings,
    key: str,
    job_id: UUID,
    job_dir: Path,
    pdf_path: Path,
    renders_dir: Path,
    page_number: int,
    omissions: object,
    options: JobOptions,
    emit: Emit,
    total: int,
) -> list[FigurePayload]:
    """Reinterpret one page's figures as Mermaid (primary capability).

    Each figure is cropped, optionally grounded in native text inside the bbox,
    sent to the Cloud converter, validated, and (by default) vision-verified.
    The resulting DiagramResult rides on the FigurePayload so
    resolve_page_figures can emit the tiered representation. Disabled or
    failed conversions leave `diagram=None` → image fallback.
    """
    payloads = _payloads(omissions)
    if not payloads or not options.diagram_to_mermaid:
        return payloads
    from src.pdf import native_text_in_bbox  # bbox-clipped grounding for figures

    converter = GlmFlashAgent(
        settings.OLLAMA_CLOUD_URL,
        key,
        model=settings.AGENT_MODEL,
        thinking_effort=options.thinking_diagram,
        timeout=settings.AGENT_TIMEOUT_SECONDS,
        system_prompt=DIAGRAM_TEMPLATE,
    )
    verifier = (
        GlmFlashAgent(
            settings.OLLAMA_CLOUD_URL,
            key,
            model=settings.AGENT_MODEL,
            thinking_effort=options.thinking_diagram,
            timeout=settings.AGENT_TIMEOUT_SECONDS,
            system_prompt=DIAGRAM_VERIFY_TEMPLATE,
        )
        if options.diagram_verify
        else None
    )
    crops_dir = job_dir / "figures"
    crops_dir.mkdir(parents=True, exist_ok=True)
    allowed_types = settings.allowed_diagram_types()
    results: list[FigurePayload] = []
    for payload in payloads:
        result = await _convert_one(
            converter,
            verifier,
            pdf_path=pdf_path,
            renders_dir=renders_dir,
            crops_dir=crops_dir,
            page_number=page_number,
            payload=payload,
            native_text_fn=native_text_in_bbox,
            options=options,
            allowed_types=allowed_types,
        )
        results.append(
            FigurePayload(
                index=payload.index,
                bbox=payload.bbox,
                alt=payload.alt,
                caption=payload.caption,
                diagram=result,
            )
        )
        await emit(
            {
                "event": "log",
                "job_id": str(job_id),
                "level": "info",
                "message": (
                    f"page {page_number}/{total} figure {payload.index}: "
                    + (
                        f"mermaid {result.diagram_type} (confidence {result.confidence})"
                        if result.convertible
                        else f"image fallback ({result.reason})"
                    )
                ),
            }
        )
    return results


async def _convert_one(
    converter: GlmFlashAgent,
    verifier: GlmFlashAgent | None,
    *,
    pdf_path: Path,
    renders_dir: Path,
    crops_dir: Path,
    page_number: int,
    payload: FigurePayload,
    native_text_fn: Callable[[Path, int, tuple[float, float, float, float], Path | None], str],
    options: JobOptions,
    allowed_types: frozenset[str],
) -> DiagramResult:
    """Crop one figure + convert it; any failure degrades to image fallback."""
    from src.pdf import crop_from_render, render_filename

    render_path = renders_dir / render_filename(page_number)
    x0, y0, x1, y1 = (round(v) for v in payload.bbox)
    crop_path = crops_dir / f"page-{page_number:03d}-fig-{payload.index:02d}.png"
    try:
        data = crop_from_render(render_path, (x0, y0, x1, y1))
        crop_path.write_bytes(data)
    except ValueError, OSError:
        return DiagramResult(convertible=False, reason="figure crop failed")
    try:
        grounding = native_text_fn(pdf_path, page_number, payload.bbox, render_path)
    except Exception:  # noqa: BLE001 - grounding is best-effort
        grounding = ""
    return await convert_figure(
        converter,
        verifier,
        crop_path,
        grounding,
        page_number=page_number,
        width=max(0, x1 - x0),
        height=max(0, y1 - y0),
        allowed_types=allowed_types,
        min_confidence=options.diagram_min_confidence,
        verify=options.diagram_verify,
    )


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
