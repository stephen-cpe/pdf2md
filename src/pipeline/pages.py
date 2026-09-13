"""Per-page pipeline — transcribe, verify, orchestrate.

transcribe: render → OCR → agent (+1 corrective retry on malformed
    envelope) → checkpoint. Terminal pages are skipped, never re-run.
verify: A.3 verdict per attempt; miss-list + DPI bump per retry;
    needs_review after MAX_PAGE_RETRIES (never silent, never blocking).
orchestrate: sequential loop, checkpoint every page, rolling context +
    outline with DB preload on resume. PauseJob → job paused, resumable.
look-ahead: OCR(N+1) overlaps transcribe(N); byte-identical output.

All model calls go through resilient(). Render defaults to the pdf
renderer; tests inject fakes. Statuses persisted via repo.
"""

import asyncio
import functools
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.db import repo
from src.db.models import JobStatus, Page, PageStatus
from src.pdf import RenderInfo, native_text_if_viable, render_filename, render_page
from src.pipeline.agent import AgentResult, TranscriptionAgent
from src.pipeline.control import IllegalTransitionError, JobControl
from src.pipeline.coverage import floor_failure, recall
from src.pipeline.envelope import (
    MAX_ENVELOPE_ATTEMPTS,
    EnvelopeError,
    ParsedEnvelope,
    ParsedVerdict,
    corrective_prompt,
    parse_transcription,
    parse_verdict,
)
from src.pipeline.hashes import file_hash, page_source_hash, text_hash
from src.pipeline.ocr import OcrEngine
from src.pipeline.resilience import PauseJob, resilient

OcrCaller = OcrEngine
AgentCaller = TranscriptionAgent
RenderFn = Callable[[Path, int, int, Path], RenderInfo]
NativeTextFn = Callable[[Path, int, int], "str | None"]


@dataclass
class PageDeps:
    """Injectable stage callables + knobs (fakes in tests, adapters live)."""

    ocr: OcrCaller
    agent: AgentCaller
    verifier: AgentCaller
    render_fn: RenderFn = render_page
    render_dpi: int = 200
    dpi_step: int = 100
    max_dpi: int = 400
    coverage_threshold: int = 95
    coverage_floor: float = 0.80
    coverage_floor_min_ocr_tokens: int = 30
    max_page_retries: int = 2
    rolling_context_pages: int = 3
    # Per-page reference routing (driver passes the setting; the dataclass
    # default stays OCR-first so direct/test construction is unchanged).
    native_text_first: bool = False
    native_text_min_words: int = 20
    native_text_fn: NativeTextFn = native_text_if_viable
    # Master OCR switch. False = never call the OCR model: pages without a
    # viable native layer transcribe vision-only with an empty reference.
    ocr_enabled: bool = True


@dataclass
class PageOutcome:
    """What one process_page call did."""

    page_number: int
    status: PageStatus
    retries_used: int = 0
    dpi_used: int = 200
    coverage: int | None = None
    markdown: str = ""
    skipped: bool = False


@dataclass
class RunSummary:
    """One run_pages pass over the outstanding range."""

    done_pages: list[int] = field(default_factory=list)
    stopped: str = "completed"  # completed | cancelled | paused


PageHook = Callable[[PageOutcome], Awaitable[None]]
PageStartHook = Callable[[int], Awaitable[None]]


def headings_of(markdown: str) -> list[str]:
    """Outline atoms: ATX headings in document order."""
    return [line.strip() for line in markdown.splitlines() if line.lstrip().startswith("#")]


async def _existing_terminal(session: AsyncSession, job_id: UUID, page_number: int) -> Page | None:
    from sqlalchemy import select

    page = await session.scalar(
        select(Page).where(Page.job_id == job_id, Page.page_number == page_number)
    )
    if page is not None and page.status in repo.TERMINAL_PAGE_STATUSES:
        return page
    return None


async def _transcribe_with_corrective(
    agent: AgentCaller,
    image: Path,
    ocr_text: str,
    width: int,
    height: int,
    page_number: int,
    context: str,
    outline: str,
) -> tuple[ParsedEnvelope, AgentResult, str]:
    """Agent call + up to 1 corrective retry; returns envelope, raw, extra-ms note.

    Raises EnvelopeError (with verbatim raw) when the cap is hit — the caller
    preserves the raw response and degrades to needs_review (A.2/§8).
    """
    extra = ""
    last_error: EnvelopeError | None = None
    for _ in range(MAX_ENVELOPE_ATTEMPTS):
        result: AgentResult = await resilient(
            functools.partial(
                agent.transcribe,
                image,
                ocr_text,
                width,
                height,
                page_number,
                context + extra,
                outline,
            ),
            operation="transcribe",
        )
        try:
            return parse_transcription(result.raw), result, extra
        except EnvelopeError as exc:
            last_error = exc
            extra = (
                "\n\nCORRECTIVE INSTRUCTION (response format only, not content): "
                + corrective_prompt(exc)
            )
    assert last_error is not None
    raise last_error


async def _render_native(
    pdf_path: Path,
    page_number: int,
    render_path: Path,
    deps: PageDeps,
) -> tuple[str, int, int, int] | None:
    """Render at base DPI and use the native text layer as the reference.

    Returns (reference_text, width, height, dpi) when the page has a
    substantial native text layer and NATIVE_TEXT_FIRST is enabled; None when
    the caller should take the OCR path. The reference feeds the same prompt
    and coverage floor as OCR text — it is exact, costs zero model calls, and
    is never a lossy re-reading (the FR-FLR rationale favors it whenever the
    layer exists).
    """
    if not deps.native_text_first:
        return None
    reference = deps.native_text_fn(pdf_path, page_number, deps.native_text_min_words)
    if reference is None:
        return None
    info = await asyncio.to_thread(
        deps.render_fn, pdf_path, page_number, deps.render_dpi, render_path
    )
    return reference, info.width, info.height, deps.render_dpi


async def process_page(
    session: AsyncSession,
    *,
    job_id: UUID,
    page_number: int,
    pdf_path: Path,
    renders_dir: Path,
    deps: PageDeps,
    rolling_context: str = "",
    outline: str = "",
) -> PageOutcome:
    """Transcribe + verify one page with retries; checkpoint the outcome."""
    terminal = await _existing_terminal(session, job_id, page_number)
    if terminal is not None:
        return PageOutcome(
            page_number=page_number,
            status=terminal.status,
            markdown=terminal.markdown or "",
            skipped=True,
        )
    render_path = renders_dir / render_filename(page_number)
    start = time.perf_counter()
    native = await _render_native(pdf_path, page_number, render_path, deps)
    if native is not None:
        ocr_text, width, height, actual_dpi = native
        first_ms = (time.perf_counter() - start) * 1000.0
        reference_kind = "native"
    else:
        ocr_text, width, height, actual_dpi = await _render_ocr(
            pdf_path, page_number, deps.render_dpi, render_path, deps
        )
        first_ms = (time.perf_counter() - start) * 1000.0
        reference_kind = "ocr" if deps.ocr_enabled else "vision"
    return await _cycles(
        session,
        job_id=job_id,
        page_number=page_number,
        pdf_path=pdf_path,
        render_path=render_path,
        ocr_text=ocr_text,
        width=width,
        height=height,
        dpi=actual_dpi,
        ocr_ms_base=first_ms,
        deps=deps,
        rolling_context=rolling_context,
        outline=outline,
        reference_kind=reference_kind,
    )


async def _load_prior_context(
    session: AsyncSession, job_id: UUID, keep: int
) -> tuple[list[str], list[str]]:
    """Resume continuity: last N verified markdowns + all headings so far."""
    from sqlalchemy import select

    rows = (
        await session.scalars(
            select(Page.markdown)
            .where(Page.job_id == job_id, Page.status.in_(repo.TERMINAL_PAGE_STATUSES))
            .order_by(Page.page_number)
        )
    ).all()
    texts = [text for text in rows if text]
    context = texts[-keep:] if keep > 0 else []
    outline = [heading for text in texts for heading in headings_of(text)]
    return context, outline


async def _run_loop(
    factory: async_sessionmaker[AsyncSession],
    *,
    job_id: UUID,
    pdf_path: Path,
    renders_dir: Path,
    deps: PageDeps,
    control: JobControl,
    lookahead: bool,
    on_page: PageHook | None = None,
    on_page_start: PageStartHook | None = None,
) -> RunSummary:
    async with factory() as session:
        job = await repo.get_job(session, job_id)
        if job is None:
            raise KeyError(f"unknown job {job_id}")
        start = await repo.find_resume_point(session, job_id)
        if start is None:
            return RunSummary()
        context_list, outline_list = await _load_prior_context(
            session, job_id, deps.rolling_context_pages
        )
    context: deque[str] = deque(context_list, maxlen=deps.rolling_context_pages or None)
    summary = RunSummary()

    async def _ocr_for(page_number: int) -> tuple[str, RenderInfo, Path]:
        path = renders_dir / render_filename(page_number)
        if not deps.ocr_enabled:
            # Vision-only arm: render once at base DPI, empty reference.
            info = await asyncio.to_thread(
                deps.render_fn, pdf_path, page_number, deps.render_dpi, path
            )
            return "", info, path
        # use same fallback logic as _render_ocr but return RenderInfo
        last_exc: BaseException | None = None
        for try_dpi in _fallback_dpis(deps.render_dpi):
            try:
                info = await asyncio.to_thread(deps.render_fn, pdf_path, page_number, try_dpi, path)
                try:
                    ocr = await deps.ocr.run_page(path, "Text Recognition:")
                except TimeoutError as exc:
                    raise PauseJob(str(exc)) from exc
                except Exception as exc:
                    import httpx as _httpx

                    if isinstance(exc, _httpx.HTTPError):
                        ocr = await resilient(
                            functools.partial(deps.ocr.run_page, path, "Text Recognition:"),
                            operation="ocr",
                        )
                    else:
                        raise
                if _is_degenerate_ocr(ocr.text):
                    raise ValueError(f"degenerate OCR at {try_dpi} DPI")
                return ocr.text, info, path
            except (PauseJob, TimeoutError, ValueError) as exc:
                is_timeout = isinstance(exc, (TimeoutError, ValueError))
                if isinstance(exc, PauseJob):
                    cause = exc.__cause__ or exc.__context__
                    import httpx as _httpx2b

                    if (
                        isinstance(cause, (_httpx2b.TimeoutException, TimeoutError))
                        or "degenerate" in str(exc)
                        or "timed out" in str(exc).lower()
                    ):
                        is_timeout = True
                    else:
                        raise
                if not is_timeout:
                    raise
                last_exc = exc
                await _unload_ocr(deps)
                await asyncio.sleep(1)
                continue
        # final pymupdf fallback — only for timeout/degenerate, not network outage
        if isinstance(last_exc, PauseJob):
            cause = last_exc.__cause__ or last_exc.__context__
            import httpx as _httpx3b

            if (
                not isinstance(cause, (_httpx3b.TimeoutException, TimeoutError))
                and "degenerate" not in str(last_exc)
                and "timed out" not in str(last_exc).lower()
            ):
                raise last_exc
        try:
            import pymupdf as _pym

            with _pym.open(pdf_path) as _doc:
                _txt = _doc[page_number - 1].get_text().strip()
            if _txt:
                info = await asyncio.to_thread(deps.render_fn, pdf_path, page_number, 72, path)
                return _txt, info, path
        except Exception:  # noqa: BLE001, S110
            pass
        if last_exc is not None:
            raise last_exc
        info = await asyncio.to_thread(deps.render_fn, pdf_path, page_number, 72, path)
        return "", info, path

    pending_ocr = None
    if lookahead:
        pending_ocr = asyncio.ensure_future(_ocr_for(start))

    for page_number in range(start, job.page_count + 1):
        await control.wait_if_paused()
        if control.cancel_requested:
            async with factory() as session:
                await repo.set_job_status(session, job_id, JobStatus.CANCELLED, "transcribing")
            summary.stopped = "cancelled"
            return summary
        if on_page_start is not None:
            await on_page_start(page_number)
        try:
            if lookahead:
                assert pending_ocr is not None
                ocr_text, _, render_path = await pending_ocr
                if page_number + 1 <= job.page_count:
                    pending_ocr = asyncio.ensure_future(_ocr_for(page_number + 1))
                # Transcribe+verify inline (per-page part of process_page).
                async with factory() as session:
                    outcome = await _transcribe_verify_only(
                        session,
                        job_id=job_id,
                        page_number=page_number,
                        pdf_path=pdf_path,
                        render_path=render_path,
                        ocr_text=ocr_text,
                        deps=deps,
                        rolling_context="\n\n".join(context),
                        outline="\n".join(outline_list),
                    )
            else:
                async with factory() as session:
                    outcome = await process_page(
                        session,
                        job_id=job_id,
                        page_number=page_number,
                        pdf_path=pdf_path,
                        renders_dir=renders_dir,
                        deps=deps,
                        rolling_context="\n\n".join(context),
                        outline="\n".join(outline_list),
                    )
        except PauseJob:
            async with factory() as session:
                await repo.set_job_status(session, job_id, JobStatus.PAUSED, "transcribing")
            summary.stopped = "paused"
            return summary
        summary.done_pages.append(page_number)
        if on_page is not None:
            await on_page(outcome)
        if outcome.status in repo.TERMINAL_PAGE_STATUSES and outcome.markdown:
            context.append(outcome.markdown)
            outline_list.extend(headings_of(outcome.markdown))
    return summary


async def _transcribe_verify_only(
    session: AsyncSession,
    *,
    job_id: UUID,
    page_number: int,
    pdf_path: Path,
    render_path: Path,
    ocr_text: str,
    deps: PageDeps,
    rolling_context: str,
    outline: str,
) -> PageOutcome:
    """Transcribe+verify for a look-ahead page (render+OCR prefetched at base DPI)."""
    width, height = _render_dims(render_path)
    return await _cycles(
        session,
        job_id=job_id,
        page_number=page_number,
        pdf_path=pdf_path,
        render_path=render_path,
        ocr_text=ocr_text,
        width=width,
        height=height,
        dpi=deps.render_dpi,
        deps=deps,
        rolling_context=rolling_context,
        outline=outline,
        reference_kind="vision" if not deps.ocr_enabled else "ocr",
    )


def _render_dims(render_path: Path) -> tuple[int, int]:
    import struct

    data = Path(render_path).read_bytes()
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def _is_degenerate_ocr(text: str) -> bool:
    """Detect glm-ocr degenerate loop (infinite ``` fences).

    High-resolution renders of dense code listings (e.g. OCP p4 at 84+ DPI)
    trigger the model to emit endless repetitive fences that never reach
    done. The stream trickles so httpx idle timeout never fires.
    Detect via excessive fence markers.
    """
    if text.count("```") > 25:
        return True
    if text.count("```\n```") > 10:
        return True
    # long output with low variety: same 30-char window repeated
    if len(text) > 4000:
        # check for a 20-char substring that repeats >30 times
        window = text[:20]
        if window and text.count(window) > 30:
            return True
    return False


def _fallback_dpis(requested: int) -> list[int]:
    """Descending fallback DPIs to try when OCR hangs/loops at requested DPI.

    Observed cliff: 80 OK, 84 fails, 100 fails, 200 fails. Jumping
    straight to 80 after requested avoids wasting 120s×3 retries at
    intermediate DPIs that also hang. 72 is final fallback (smallest
    allowed, still legible for code at 72 we verified).
    """
    if requested <= 80:
        return [requested]
    # try requested, then known-good 80, then 72
    candidates = [requested, 80, 72]
    seen: set[int] = set()
    out: list[int] = []
    for dpi in candidates:
        if dpi not in seen:
            seen.add(dpi)
            out.append(dpi)
    return out


async def _unload_ocr(deps: PageDeps) -> None:
    """Best-effort unload of glm-ocr after a hang so next DPI isn't queued behind it."""
    try:
        import httpx as _httpx

        url = getattr(deps.ocr, "_local_url", None)
        model = getattr(deps.ocr, "_model", "glm-ocr")
        if url:
            async with _httpx.AsyncClient(timeout=5) as _client:
                await _client.post(f"{url}/api/generate", json={"model": model, "keep_alive": 0})
        else:
            # fallback hardcoded local
            async with _httpx.AsyncClient(timeout=5) as _client:
                await _client.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "glm-ocr", "keep_alive": 0},
                )
    except Exception:  # noqa: BLE001, S110
        pass


async def _render_ocr(
    pdf_path: Path,
    page_number: int,
    dpi: int,
    render_path: Path,
    deps: PageDeps,
) -> tuple[str, int, int, int]:
    """Render + OCR one page; returns (ocr_text, width, height, actual_dpi).

    When the OCR stage is disabled (ablation / OCR-free arm), this renders
    once at the requested DPI and returns an empty reference — vision-only,
    zero local model calls. The caller records reference "vision".
    Otherwise the full ladder below applies: tries lower DPIs on OCR
    timeout/degenerate output. High-res dense pages can cause glm-ocr to
    enter an infinite fence loop that never triggers the httpx idle timeout
    (token trickle). Detected via _is_degenerate_ocr and wall-clock wait_for
    in GlmOcrEngine, then retried at a lower DPI. Final fallback: pymupdf
    text extraction (born-digital path) so the job never hangs forever
    (NFR-1 never silently drops a page; hang is worse than fallback).
    """
    if not deps.ocr_enabled:
        info = await asyncio.to_thread(deps.render_fn, pdf_path, page_number, dpi, render_path)
        return "", info.width, info.height, dpi
    last_exc: BaseException | None = None
    for try_dpi in _fallback_dpis(dpi):
        try:
            info = await asyncio.to_thread(
                deps.render_fn, pdf_path, page_number, try_dpi, render_path
            )
            # OCR with fast fallback on timeout/degenerate, but 5xx/429 still retry
            # via resilient. TimeoutError is wall-clock from GlmOcrEngine.wait_for.
            try:
                ocr = await deps.ocr.run_page(render_path, "Text Recognition:")
            except TimeoutError as exc:
                raise PauseJob(str(exc)) from exc
            except Exception as exc:
                # httpx 5xx/429/Connect etc: retry at same DPI via resilient
                import httpx as _httpx

                if isinstance(exc, _httpx.HTTPError):
                    # let resilient handle retries; if it raises PauseJob, fall through to next DPI
                    ocr = await resilient(
                        functools.partial(deps.ocr.run_page, render_path, "Text Recognition:"),
                        operation="ocr",
                    )
                else:
                    raise
            if _is_degenerate_ocr(ocr.text):
                raise ValueError(
                    f"degenerate OCR output at {try_dpi} DPI (fence loop, len={len(ocr.text)})"
                )
            # success: return with the DPI that worked (info reflects try_dpi)
            return ocr.text, info.width, info.height, try_dpi
        except (PauseJob, TimeoutError, ValueError) as exc:
            # Only fallback to lower DPI for timeout/degenerate, not for network 5xx/ConnectError
            # (those should pause the job per FR-AGT-8, not silently degrade to pymupdf).
            is_timeout = isinstance(exc, (TimeoutError, ValueError))
            if isinstance(exc, PauseJob):
                cause = exc.__cause__ or exc.__context__
                # httpx timeout is retryable as fallback, but ConnectError/5xx is not

                if (
                    isinstance(cause, (httpx.TimeoutException, TimeoutError))
                    or isinstance(cause, ValueError)
                    and "degenerate" in str(exc)
                ):
                    is_timeout = True
                else:
                    # network 5xx/ConnectError: pause, don't fallback
                    raise
            if not is_timeout and isinstance(exc, ValueError):
                is_timeout = True
            if not is_timeout:
                raise
            last_exc = exc
            # unload hanging model before next DPI so it isn't queued behind previous generation
            await _unload_ocr(deps)
            # small backoff so Ollama can reclaim VRAM
            await asyncio.sleep(1)
            # try next lower DPI; keep render_path overwritten on next loop
            continue
        except Exception:
            raise
    # All DPI attempts failed: for network outage, propagate pause (don't hide with pymupdf)
    if isinstance(last_exc, PauseJob):
        cause = last_exc.__cause__ or last_exc.__context__
        import httpx as _httpx3

        # TimeoutException and degenerate ValueError are fallback-eligible; ConnectError/5xx are not
        is_timeout_cause = (
            isinstance(cause, (_httpx3.TimeoutException, TimeoutError))
            or "degenerate" in str(last_exc)
            or "timed out" in str(last_exc).lower()
        )
        if not is_timeout_cause:
            raise last_exc
    try:
        import pymupdf

        with pymupdf.open(pdf_path) as doc:
            txt = doc[page_number - 1].get_text().strip()
        if txt:
            # render at 72 for image dims so agent still gets an image
            info = await asyncio.to_thread(deps.render_fn, pdf_path, page_number, 72, render_path)
            return txt, info.width, info.height, 72
    except Exception:  # noqa: BLE001, S110
        pass
    # re-raise last OCR error to let caller handle (will become needs_review via PauseJob)
    if last_exc is not None:
        raise last_exc
    # fallback empty (should not happen)
    info = await asyncio.to_thread(deps.render_fn, pdf_path, page_number, 72, render_path)
    return "", info.width, info.height, 72


def _telemetry(
    render_path: Path,
    width: int,
    height: int,
    model: str | None,
    effort: str | None,
    ocr_ms: float,
    transcribe_ms: float,
    verify_ms: float,
    verification_score: int,
) -> dict[str, object]:
    """FR-AGT-10 per-page telemetry payload for pages.timings."""
    try:
        image_bytes: int | None = render_path.stat().st_size
    except OSError:
        image_bytes = None
    return {
        "image_width": width,
        "image_height": height,
        "image_bytes": image_bytes,
        "model": model,
        "thinking_effort": effort,
        "ocr_ms": round(ocr_ms, 1),
        "transcribe_ms": round(transcribe_ms, 1),
        "verify_ms": round(verify_ms, 1),
        "verification_score": verification_score,
    }


async def _cycles(
    session: AsyncSession,
    *,
    job_id: UUID,
    page_number: int,
    pdf_path: Path,
    render_path: Path,
    ocr_text: str,
    width: int,
    height: int,
    dpi: int,
    deps: PageDeps,
    rolling_context: str,
    outline: str,
    ocr_ms_base: float = 0.0,
    reference_kind: str = "ocr",
) -> PageOutcome:
    """Shared verify-retry core (4.2).

    Retries re-render at higher DPI (FR-PDF-4). A native reference is kept
    verbatim on retry while only the render sharpens; a vision-only page
    (OCR disabled, no native layer) simply re-renders with its empty
    reference. OCR pages re-OCR the new render (FR-OCR-2).
    """
    misses_note = ""
    best_markdown = ""
    best_coverage: int | None = None
    best_env: ParsedEnvelope | None = None
    best_model: str | None = None
    best_effort: str | None = None
    best_floor: float | None = None
    best_floor_failed = False
    prompt_tokens = completion_tokens = 0
    ocr_ms = ocr_ms_base
    transcribe_ms = verify_ms = 0.0
    outcome_dpi = dpi
    outcome_width, outcome_height = width, height
    for attempt in range(deps.max_page_retries + 1):
        if attempt > 0:
            # An agentic retry: (re-)render at the next DPI step.
            outcome_dpi = min(deps.max_dpi, outcome_dpi + deps.dpi_step)
            start = time.perf_counter()
            if reference_kind == "ocr":
                ocr_text, outcome_width, outcome_height, actual_dpi = await _render_ocr(
                    pdf_path, page_number, outcome_dpi, render_path, deps
                )
                # actual_dpi may be lower than outcome_dpi if fallback triggered
                outcome_dpi = actual_dpi
                ocr_ms += (time.perf_counter() - start) * 1000.0
                width, height = outcome_width, outcome_height
            else:
                # Native or vision-only: keep the reference, sharpen the image.
                info = await asyncio.to_thread(
                    deps.render_fn, pdf_path, page_number, outcome_dpi, render_path
                )
                outcome_width, outcome_height = info.width, info.height
                ocr_ms += (time.perf_counter() - start) * 1000.0
                width, height = outcome_width, outcome_height
        context = (rolling_context + misses_note).strip()
        start = time.perf_counter()
        try:
            env, agent_result, _ = await _transcribe_with_corrective(
                deps.agent, render_path, ocr_text, width, height, page_number, context, outline
            )
        except EnvelopeError as exc:
            best_markdown, best_coverage, best_env = "", 0, None
            misses_note = f"\n\nPrevious attempt malformed ({exc.detail}); re-emit carefully."
            if attempt >= deps.max_page_retries:
                break
            continue
        transcribe_ms += (time.perf_counter() - start) * 1000.0
        prompt_tokens += agent_result.prompt_tokens or 0
        completion_tokens += agent_result.completion_tokens or 0
        best_markdown, best_env = env.markdown, env
        best_model, best_effort = agent_result.model, agent_result.thinking_effort
        candidate = f"OCR REFERENCE:\n{ocr_text}\nCANDIDATE MARKDOWN:\n{env.markdown}"
        start = time.perf_counter()
        verdict_raw = await resilient(
            functools.partial(
                deps.verifier.transcribe, render_path, candidate, width, height, page_number, "", ""
            ),
            operation="verify",
        )
        verify_ms += (time.perf_counter() - start) * 1000.0
        prompt_tokens += verdict_raw.prompt_tokens or 0
        completion_tokens += verdict_raw.completion_tokens or 0
        try:
            verdict = parse_verdict(verdict_raw.raw)
        except EnvelopeError:
            verdict = ParsedVerdict(coverage=0, misses=("unparseable verdict",))
        best_coverage = verdict.coverage
        floor_score = recall(ocr_text, env.markdown, deps.coverage_floor_min_ocr_tokens)
        floor_reason = floor_failure(floor_score, deps.coverage_floor)
        best_floor = floor_score
        best_floor_failed = floor_reason is not None
        if (
            verdict.verdict == "pass"
            and verdict.coverage >= deps.coverage_threshold
            and floor_reason is None
        ):
            figures = [
                {"index": f.index, "bbox": list(f.bbox), "alt": f.alt, "caption": f.caption}
                for f in env.figures
            ]
            await repo.checkpoint_page(
                session,
                job_id,
                page_number,
                status=PageStatus.VERIFIED,
                render_dpi=outcome_dpi,
                markdown=env.markdown,
                coverage_score=verdict.coverage,
                retries=attempt,
                needs_review=False,
                omissions={
                    "figures": figures,
                    "furniture": {
                        "header": env.furniture.header,
                        "footer": env.furniture.footer,
                        "page_number": env.furniture.page_number,
                    },
                    "notes": env.notes,
                    "floor_score": floor_score,
                    "reference": reference_kind,
                },
                token_usage={"prompt": prompt_tokens, "completion": completion_tokens},
                timings=_telemetry(
                    render_path,
                    outcome_width,
                    outcome_height,
                    best_model,
                    best_effort,
                    ocr_ms,
                    transcribe_ms,
                    verify_ms,
                    verdict.coverage,
                ),
                source_page_hash=page_source_hash(pdf_path, page_number),
                render_hash=file_hash(render_path),
                ocr_hash=text_hash(ocr_text),
                markdown_hash=text_hash(env.markdown),
            )
            return PageOutcome(
                page_number=page_number,
                status=PageStatus.VERIFIED,
                retries_used=attempt,
                dpi_used=outcome_dpi,
                coverage=verdict.coverage,
                markdown=env.markdown,
            )
        misses_note = f"\n\nVerification misses to fix: {list(verdict.misses)}"
        if floor_reason is not None:
            misses_note = (
                f"\n\nDeterministic coverage floor FAILED ({floor_reason}). "
                "Re-transcribe the COMPLETE page verbatim; do not omit or "
                "paraphrase any source text."
            )
        if attempt >= deps.max_page_retries:
            break
    figures = (
        [
            {"index": f.index, "bbox": list(f.bbox), "alt": f.alt, "caption": f.caption}
            for f in best_env.figures
        ]
        if best_env is not None
        else []
    )
    await repo.checkpoint_page(
        session,
        job_id,
        page_number,
        status=PageStatus.NEEDS_REVIEW,
        render_dpi=outcome_dpi,
        markdown=best_markdown,
        coverage_score=best_coverage or 0,
        retries=deps.max_page_retries,
        needs_review=True,
        omissions={
            "figures": figures,
            "notes": (
                "coverage floor failed: judge score alone is insufficient"
                if best_floor_failed
                else "verification cap reached"
            ),
            "floor_score": best_floor,
            "reference": reference_kind,
        },
        token_usage={"prompt": prompt_tokens, "completion": completion_tokens},
        timings=_telemetry(
            render_path,
            outcome_width,
            outcome_height,
            best_model,
            best_effort,
            ocr_ms,
            transcribe_ms,
            verify_ms,
            best_coverage or 0,
        ),
        source_page_hash=page_source_hash(pdf_path, page_number),
        render_hash=file_hash(render_path),
        ocr_hash=text_hash(ocr_text),
        markdown_hash=text_hash(best_markdown),
    )
    return PageOutcome(
        page_number=page_number,
        status=PageStatus.NEEDS_REVIEW,
        retries_used=deps.max_page_retries,
        dpi_used=outcome_dpi,
        coverage=best_coverage or 0,
        markdown=best_markdown,
    )


async def run_pages(
    factory: async_sessionmaker[AsyncSession],
    *,
    job_id: UUID,
    pdf_path: Path,
    renders_dir: Path,
    deps: PageDeps,
    control: JobControl,
    on_page: PageHook | None = None,
    on_page_start: PageStartHook | None = None,
) -> RunSummary:
    """Sequential page loop (4.3): checkpoint every page, honor signals."""
    async with factory() as session:
        await repo.set_job_status(session, job_id, JobStatus.TRANSCRIBING, "transcribing")
    if control.status not in (JobStatus.TRANSCRIBING, JobStatus.PAUSED):
        try:
            control.transition(JobStatus.TRANSCRIBING)
        except IllegalTransitionError:
            pass  # resume into an already-transcribing control
    return await _run_loop(
        factory,
        job_id=job_id,
        pdf_path=pdf_path,
        renders_dir=renders_dir,
        deps=deps,
        control=control,
        lookahead=False,
        on_page=on_page,
        on_page_start=on_page_start,
    )


async def run_pages_lookahead(
    factory: async_sessionmaker[AsyncSession],
    *,
    job_id: UUID,
    pdf_path: Path,
    renders_dir: Path,
    deps: PageDeps,
    control: JobControl,
    on_page: PageHook | None = None,
    on_page_start: PageStartHook | None = None,
) -> RunSummary:
    """OCR(N+1) overlaps transcribe(N) (4.4); output identical to sequential."""
    async with factory() as session:
        await repo.set_job_status(session, job_id, JobStatus.TRANSCRIBING, "transcribing")
    if control.status not in (JobStatus.TRANSCRIBING, JobStatus.PAUSED):
        try:
            control.transition(JobStatus.TRANSCRIBING)
        except IllegalTransitionError:
            pass  # resume into an already-transcribing control
    return await _run_loop(
        factory,
        job_id=job_id,
        pdf_path=pdf_path,
        renders_dir=renders_dir,
        deps=deps,
        control=control,
        lookahead=True,
        on_page=on_page,
        on_page_start=on_page_start,
    )


__all__ = [
    "AgentCaller",
    "OcrCaller",
    "PageDeps",
    "PageHook",
    "PageOutcome",
    "PageStartHook",
    "RenderFn",
    "RunSummary",
    "headings_of",
    "process_page",
    "run_pages",
    "run_pages_lookahead",
]
