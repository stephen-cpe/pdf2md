"""REST + WebSocket API — §5.1/§5.2.

Single-job guard (FR-JOB-3): one running job; new uploads get 409 while busy.
Errors are RFC 9457 problem+json. The heavy lifting lives in
pipeline.driver; routes only validate, register, and report.
"""

import asyncio
import contextlib
import json
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select
from starlette.websockets import WebSocketDisconnect

from src import health as health_mod
from src.api.events import BusHub, EventBus
from src.config import Settings, load_settings
from src.db import engine as engine_mod
from src.db import repo
from src.db.models import Event, Job, JobStatus
from src.pdf import preflight
from src.pipeline import driver as driver_mod
from src.pipeline.control import IllegalTransitionError, JobControl
from src.pipeline.driver import JobOptions
from src.pipeline.report import build_report, collect_job_images, collect_job_pages
from src.workspace import OutputDirError, Workspace

PROBLEM_JSON = "application/problem+json"


async def _next_event(stream: AsyncGenerator[dict[str, Any]]) -> dict[str, Any]:
    """Advance a subscription by one event (create_task needs a coroutine)."""
    return await stream.__anext__()


# Idle keepalive: pages can run 20+ min with zero bus events (local OCR on
# CPU), and idle WebSockets get reaped by browsers/intermediaries. A ping
# every interval keeps the socket alive and lets the client detect death.
# Must stay well under common 60s idle timeouts.
WS_HEARTBEAT_SECONDS = 20.0


async def _stream_live_events(bus: EventBus, websocket: WebSocket, job_id: str = "") -> None:
    """Forward live bus events until the client disconnects.

    Races the subscription against websocket.receive(): the protocol
    queues a disconnect message on shutdown/connection loss, so the
    handler always notices death. Parking in queue.get() alone never
    observes it — the connection task then blocks server shutdown
    forever ("Waiting for background tasks"), forcing a second Ctrl+C
    down the force-quit path. Disconnect is a normal return (the replay
    buffer retains history); genuine cancellation propagates to the
    caller (ws_job converts it to a normal exit — see below).

    Long OCR/QA gaps emit nothing, so an idle socket would look dead to
    intermediaries. On wait timeout a lightweight {"event": "ping"}
    keepalive is sent instead (clients ignore it); a reconnecting client
    replays missed events from the buffer.
    """
    stream = bus.subscribe()
    get_task: asyncio.Task[dict[str, Any]] | None = None
    recv_task: asyncio.Task[Any] | None = None
    try:
        while True:
            if get_task is None:
                get_task = asyncio.create_task(_next_event(stream))
            if recv_task is None:
                recv_task = asyncio.create_task(websocket.receive())
            done, _pending = await asyncio.wait(
                {get_task, recv_task},
                timeout=WS_HEARTBEAT_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                # Idle gap (long OCR page): keepalive so the socket is not
                # reaped; a dead client surfaces here as a send error.
                try:
                    await websocket.send_json({"event": "ping", "job_id": job_id})
                except WebSocketDisconnect, RuntimeError:
                    return
                continue
            if recv_task in done:
                try:
                    recv_task.result()
                except WebSocketDisconnect, RuntimeError:
                    pass
                return
            try:
                event = get_task.result()
            except StopAsyncIteration:
                return  # subscription ended (never happens, but be safe)
            get_task = None
            recv_task.cancel()
            await asyncio.gather(recv_task, return_exceptions=True)
            recv_task = None
            try:
                await websocket.send_json(event)
            except WebSocketDisconnect, RuntimeError:
                return  # died while sending; same as a disconnect
    finally:
        pending = [task for task in (get_task, recv_task) if task is not None and not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await stream.aclose()


def problem(status: int, title: str, detail: str) -> JSONResponse:
    """RFC 9457 problem-details response."""
    return JSONResponse(
        status_code=status,
        content={"type": "about:blank", "title": title, "status": status, "detail": detail},
        media_type=PROBLEM_JSON,
    )


class JobCreated(BaseModel):
    """POST /jobs response."""

    job_id: str


def safe_doc_stem(filename: str) -> str:
    """Filesystem-safe deliverable stem shared by upload and fallbacks."""
    stem = Path(filename or "upload.pdf").stem
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem) or "upload"


def _job_or_404(app: FastAPI, job_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"invalid job id: {job_id}") from None


async def _get_job_or_404(app: FastAPI, job_id: uuid.UUID) -> Job:
    factory = app.state.session_factory
    async with factory() as session:
        job = await repo.get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
    return job


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory (tests inject nothing — they monkeypatch driver/health)."""
    settings = settings or load_settings()
    engine = engine_mod.create_engine(settings.DATABASE_URL.get_secret_value())
    factory = engine_mod.session_factory(engine)
    workspace = Workspace(Path("./workspace"), max_retries=settings.MAX_CLEANUP_RETRIES)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        import logging as _logging

        from src.logging import configure_logging

        configure_logging()
        workspace.cleanup_deferred()
        try:
            async with factory() as session:
                jobs = await repo.job_history(session, limit=50)
                for job in jobs:
                    if job.status.value not in ("completed", "failed", "cancelled"):
                        _logging.getLogger("pdf2md").warning(
                            "incomplete job %s (%s): resume with POST /api/v1/jobs/%s/restart",
                            job.id,
                            job.status.value,
                            job.id,
                        )
        except Exception as exc:  # noqa: BLE001 - DB down must not block startup (§8 degraded mode)
            _logging.getLogger("pdf2md").warning("startup scan skipped: %s", exc)
        yield
        # Shutdown hygiene: close live sockets first so each handler's
        # disconnect race fires and its task drains (a parked handler
        # would stall "Waiting for background tasks" forever). Driver
        # tasks are cancelled; a close racing teardown may itself be
        # cancelled, which must not mask shutdown.
        for socket in list(app.state.sockets):
            with contextlib.suppress(WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
                await socket.close(code=1001)
        for task in list(app.state.tasks.values()):
            task.cancel()

    app = FastAPI(title="pdf2md", version="0.0.1")
    app.state.settings = settings
    app.state.session_factory = factory
    app.state.workspace = workspace
    app.state.buses = BusHub()
    app.state.running = {}
    app.state.tasks = {}
    app.state.sockets = set()
    app.state.job_lock = asyncio.Lock()
    app.router.lifespan_context = lifespan

    @app.exception_handler(HTTPException)
    async def _http_as_problem(request: Request, exc: HTTPException) -> JSONResponse:
        return problem(exc.status_code, "request failed", str(exc.detail))

    @app.exception_handler(IllegalTransitionError)
    async def _transition_as_problem(request: Request, exc: IllegalTransitionError) -> JSONResponse:
        return problem(409, "illegal transition", str(exc))

    @app.exception_handler(OutputDirError)
    async def _output_dir_as_problem(request: Request, exc: OutputDirError) -> JSONResponse:
        return problem(422, "invalid output directory", str(exc))

    @app.post("/api/v1/jobs", status_code=201, response_model=JobCreated)
    async def post_jobs(
        file: UploadFile = File(...),
        output_dir: str = Form(...),
        options: str = Form("{}"),
    ) -> Any:
        if not (file.filename or "").lower().endswith(".pdf"):
            return problem(422, "invalid upload", "only PDF files are accepted")
        try:
            overrides = json.loads(options or "{}")
        except ValueError as exc:
            return problem(422, "invalid options", f"options must be JSON: {exc}")
        if not isinstance(overrides, dict):
            return problem(422, "invalid options", "options must be a JSON object")
        job_options = JobOptions.from_settings(settings, overrides)
        async with app.state.job_lock:
            if app.state.running:
                return problem(409, "job running", "one job at a time; retry when idle")
            job_id = uuid.uuid4()
            job_dir = workspace.job_dir(str(job_id))
            # Keep the client's filename (sanitized to a bare stem): it names
            # the workspace copy, the deliverable, and the log lines.
            source = job_dir / f"{safe_doc_stem(file.filename or 'upload.pdf')}.pdf"
            content = await file.read()
            max_bytes = settings.MAX_PDF_MB * 1024 * 1024
            if len(content) > max_bytes:
                return problem(422, "file too large", f"cap is {settings.MAX_PDF_MB} MB")
            source.write_bytes(content)
            pre = preflight(source, max_mb=settings.MAX_PDF_MB, max_pages=settings.MAX_PDF_PAGES)
            if not pre.valid:
                return problem(422, "preflight failed", pre.error)
            try:
                out = Workspace.validate_output_dir(output_dir, create=True)
            except OutputDirError as exc:
                return problem(422, "invalid output directory", str(exc))
            from src.pipeline.prompts import PROMPT_VERSIONS, compute_pipeline_version

            snapshot = {
                "agent_model": settings.AGENT_MODEL,
                "ocr_model": settings.OCR_MODEL,
                "render_dpi": job_options.render_dpi,
                "rolling_context_pages": job_options.rolling_context_pages,
                "coverage_threshold": job_options.coverage_threshold,
                "coverage_floor": job_options.coverage_floor,
                "max_page_retries": job_options.max_page_retries,
                "thinking_transcribe": job_options.thinking_transcribe,
                "thinking_diagram": job_options.thinking_diagram,
                "toc_enabled": job_options.toc_enabled,
                "fig_details": job_options.fig_details,
                "diagram_to_mermaid": job_options.diagram_to_mermaid,
                "diagram_min_confidence": job_options.diagram_min_confidence,
                "diagram_verify": job_options.diagram_verify,
                "diagram_fallback": job_options.diagram_fallback,
                "diagram_keep_image": job_options.diagram_keep_image,
                "prompts": dict(PROMPT_VERSIONS),
            }
            pipeline_version = compute_pipeline_version(
                agent_model=settings.AGENT_MODEL,
                ocr_model=settings.OCR_MODEL,
                render_dpi=job_options.render_dpi,
                rolling_context_pages=job_options.rolling_context_pages,
                coverage_threshold=job_options.coverage_threshold,
                coverage_floor=job_options.coverage_floor,
                max_page_retries=job_options.max_page_retries,
                thinking_transcribe=job_options.thinking_transcribe,
                thinking_diagram=job_options.thinking_diagram,
                toc_enabled=job_options.toc_enabled,
                fig_details=job_options.fig_details,
                diagram_to_mermaid=job_options.diagram_to_mermaid,
                diagram_min_confidence=job_options.diagram_min_confidence,
            )
            async with factory() as session:
                import hashlib

                await repo.create_job(
                    session,
                    filename=file.filename or "upload.pdf",
                    file_sha256=hashlib.sha256(content).hexdigest(),
                    page_count=pre.page_count,
                    output_dir=str(out),
                    options=snapshot,
                    pdf_metadata=pre.metadata,
                    pipeline_version=pipeline_version,
                    job_id=job_id,
                )
            control = JobControl()
            _launch(str(job_id), source, out, job_options, control)
            return {"job_id": str(job_id)}

    def _launch(
        key: str, pdf_path: Path, out: Path, job_options: JobOptions, control: JobControl
    ) -> None:
        """Register + background a driver run (POST and restart share it)."""
        app.state.running[key] = control
        bus = app.state.buses.bus(key)

        async def _emit(event: dict[str, Any]) -> None:
            await bus.publish(event)

        task = asyncio.create_task(
            driver_mod.run_job(
                factory=factory,
                settings=settings,
                workspace=workspace,
                job_id=uuid.UUID(key),
                pdf_path=pdf_path,
                output_dir=out,
                options=job_options,
                control=control,
                emit=_emit,
            )
        )

        def _done(task: asyncio.Task[Any], keep: str = key) -> None:
            app.state.running.pop(keep, None)
            app.state.tasks.pop(keep, None)

        task.add_done_callback(_done)
        app.state.tasks[key] = task

    @app.post("/api/v1/jobs/{job_id}/restart", status_code=202)
    async def post_restart(job_id: str) -> Any:
        """Resume a stalled job (paused/cancelled/failed/crashed) on its checkpoints.

        Options come from the job's own snapshot (same pipeline,
        never a silent mix).
        """
        uid = _job_or_404(app, job_id)
        job = await _get_job_or_404(app, uid)
        async with app.state.job_lock:
            if app.state.running:
                return problem(409, "job running", "one job at a time; retry when idle")
            if job.status is JobStatus.COMPLETED:
                return problem(409, "already completed", f"job {job_id} is done")
            pdfs = sorted((workspace.job_dir(job_id)).glob("*.pdf"))
            if not pdfs:
                return problem(
                    409, "nothing to resume", f"workspace source for job {job_id} is gone"
                )
            snapshot = job.options if isinstance(job.options, dict) else {}
            job_options = JobOptions.from_settings(
                settings,
                {
                    key: snapshot[key]
                    for key in (
                        "render_dpi",
                        "coverage_threshold",
                        "coverage_floor",
                        "max_page_retries",
                        "rolling_context_pages",
                        "toc_enabled",
                        "fig_details",
                        "thinking_transcribe",
                        "thinking_diagram",
                        "diagram_to_mermaid",
                        "diagram_min_confidence",
                        "diagram_verify",
                        "diagram_fallback",
                        "diagram_keep_image",
                    )
                    if key in snapshot
                },
            )
            try:
                out = Workspace.validate_output_dir(job.output_dir, create=True)
            except OutputDirError as exc:
                return problem(422, "invalid output directory", str(exc))
            _launch(job_id, pdfs[0], out, job_options, JobControl())
            return {"job_id": job_id, "status": "restarted"}

    @app.get("/api/v1/jobs")
    async def get_jobs() -> Any:
        async with factory() as session:
            jobs = await repo.job_history(session)
        return [
            {
                "job_id": str(job.id),
                "filename": job.filename,
                "status": job.status.value,
                "stage": job.stage,
                "page_count": job.page_count,
                "created_at": job.created_at.isoformat() if job.created_at else None,
            }
            for job in jobs
        ]

    @app.get("/api/v1/jobs/{job_id}")
    async def get_job(job_id: str) -> Any:
        uid = _job_or_404(app, job_id)
        job = await _get_job_or_404(app, uid)
        done = 0
        async with factory() as session:
            _, pages = await collect_job_pages(session, uid)
            done = sum(1 for page in pages if page.status in repo.TERMINAL_PAGE_STATUSES)
        return {
            "job_id": str(job.id),
            "filename": job.filename,
            "status": job.status.value,
            "stage": job.stage,
            "page_count": job.page_count,
            "pages_done": done,
            "live": str(job.id) in app.state.running,
            "output_dir": job.output_dir,
            "options": job.options,
            "error": job.error,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        }

    @app.get("/api/v1/jobs/{job_id}/pages")
    async def get_pages(job_id: str) -> Any:
        uid = _job_or_404(app, job_id)
        await _get_job_or_404(app, uid)
        async with factory() as session:
            _, pages = await collect_job_pages(session, uid)
        return [
            {
                "page": page.page_number,
                "status": page.status.value,
                "coverage": page.coverage_score,
                "needs_review": page.needs_review,
            }
            for page in pages
        ]

    @app.get("/api/v1/jobs/{job_id}/preview")
    async def get_preview(job_id: str) -> Any:
        uid = _job_or_404(app, job_id)
        await _get_job_or_404(app, uid)
        async with factory() as session:
            _, pages = await collect_job_pages(session, uid)
        done = [
            page.markdown or ""
            for page in pages
            if page.status in repo.TERMINAL_PAGE_STATUSES and page.markdown
        ]
        return {"job_id": job_id, "markdown": "\n\n".join(done)}

    @app.get("/api/v1/jobs/{job_id}/report")
    async def get_report(job_id: str) -> Any:
        uid = _job_or_404(app, job_id)
        job = await _get_job_or_404(app, uid)
        if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
            return problem(404, "report not ready", f"job is {job.status.value}")
        async with factory() as session:
            _, pages = await collect_job_pages(session, uid)
            images = await collect_job_images(session, uid)
            results = (
                (job.options or {}).get("results", {}) if isinstance(job.options, dict) else {}
            )
            from src.pipeline.prompts import PROMPT_VERSIONS

            report = build_report(
                job,
                pages,
                pipeline_version=job.pipeline_version or "",
                prompt_versions=dict(PROMPT_VERSIONS),
                furniture_removed=results.get("furniture_removed", []),
                lint_warnings=results.get("lint_warnings", []),
                images=images,
                orphaned_figures=results.get("orphaned_figures", []),
            )
        return {"job_id": job_id, "report": report}

    @app.get("/api/v1/jobs/{job_id}/events")
    async def get_events(job_id: str) -> Any:
        uid = _job_or_404(app, job_id)
        await _get_job_or_404(app, uid)
        async with factory() as session:
            rows = (
                await session.scalars(select(Event).where(Event.job_id == uid).order_by(Event.id))
            ).all()
        return [
            {
                "ts": e.ts.isoformat() if e.ts else None,
                "level": e.level,
                "stage": e.stage,
                "message": e.message,
            }
            for e in rows
        ]

    @app.get("/api/v1/jobs/{job_id}/artifact/{name}")
    async def get_artifact(job_id: str, name: str) -> Any:
        uid = _job_or_404(app, job_id)
        job = await _get_job_or_404(app, uid)
        options = job.options if isinstance(job.options, dict) else {}
        results = options.get("results", {}) if isinstance(options, dict) else {}
        artifacts = results.get("artifacts", {}) if isinstance(results, dict) else {}
        path = artifacts.get(name)
        if not path or not Path(path).is_file():
            # Fallback when the recorded path is missing: check the
            # conventional locations in order (top-level .md, nested .md,
            # per-doc report, shared report).
            stem = safe_doc_stem(job.filename)
            candidates = {
                "document": [
                    Path(job.output_dir) / f"{stem}.md",
                    Path(job.output_dir) / stem / f"{stem}.md",
                ],
                "report": [
                    Path(job.output_dir) / stem / "conversion-report.md",
                    Path(job.output_dir) / "conversion-report.md",
                ],
            }.get(name, [])
            for fallback in candidates:
                if fallback.is_file():
                    path = str(fallback)
                    break
        if not path or not Path(path).is_file():
            return problem(404, "artifact not ready", f"no {name} artifact for job {job_id}")
        return FileResponse(path, media_type="text/markdown")

    async def _live_or_explain(job_id: str, action: str) -> JobControl:
        """Live control, or a 409 that says what actually happened + the way back.

        A missing control with an active DB status means the task ended without
        the client noticing (outage-pause, crash, or server restart): point at
        the restart endpoint instead of stranding the user.
        """
        control = cast("JobControl | None", app.state.running.get(job_id))
        if control is not None:
            return control
        uid = _job_or_404(app, job_id)
        job = await _get_job_or_404(app, uid)
        status = job.status.value
        if status == "paused":
            hint = f"job is paused; resume it with POST /api/v1/jobs/{job_id}/restart"
        elif status in ("completed", "failed", "cancelled"):
            hint = f"job already {status}; nothing to {action}"
        else:
            hint = (
                f"job shows {status} but has no live control "
                "(task ended, crashed, or server restarted); "
                f"resume with POST /api/v1/jobs/{job_id}/restart"
            )
        raise HTTPException(status_code=409, detail=f"cannot {action}: {hint}")

    @app.post("/api/v1/jobs/{job_id}/cancel")
    async def post_cancel(job_id: str) -> Any:
        (await _live_or_explain(job_id, "cancel")).request_cancel()
        return {"job_id": job_id, "status": "cancelled"}

    @app.post("/api/v1/jobs/{job_id}/pause")
    async def post_pause(job_id: str) -> Any:
        (await _live_or_explain(job_id, "pause")).request_pause()
        return {"job_id": job_id, "status": "paused"}

    @app.post("/api/v1/jobs/{job_id}/resume")
    async def post_resume(job_id: str) -> Any:
        (await _live_or_explain(job_id, "resume")).request_resume()
        return {"job_id": job_id, "status": "resumed"}

    @app.get("/api/v1/browse")
    async def browse(path: str = "") -> Any:
        """Server-side folder browser (localhost UI picker backing).

        Browsers cannot expose native directory pickers, so the UI navigates
        the server filesystem instead (this app binds 127.0.0.1; the disk
        being browsed is the operator's own). Directories only, capped.
        """
        import string

        if not path:
            drives = [
                f"{letter}:\\" for letter in string.ascii_uppercase if Path(f"{letter}:\\").is_dir()
            ]
            return {
                "path": "",
                "parent": None,
                "entries": [{"name": drive, "path": drive} for drive in drives],
            }
        candidate = Path(path)
        if not candidate.is_absolute():
            return problem(422, "invalid path", "path must be absolute")
        resolved = candidate.resolve()
        if not resolved.is_dir():
            return problem(404, "not a directory", str(resolved))
        try:
            subdirs = sorted(entry for entry in resolved.iterdir() if entry.is_dir())
        except PermissionError:
            return problem(403, "access denied", str(resolved))
        entries = [{"name": entry.name, "path": str(entry)} for entry in subdirs[:500]]
        parent = str(resolved.parent) if resolved.parent != resolved else None
        return {"path": str(resolved), "parent": parent, "entries": entries}

    @app.get("/api/v1/health")
    async def get_health() -> Any:
        import asyncio as _asyncio

        results = await _asyncio.to_thread(health_mod.run_all, settings)
        running = next(iter(app.state.running), None)
        return {
            "status": "ok" if all(r.ok for r in results) else "degraded",
            "running_job": running,
            "components": [{"name": r.name, "ok": r.ok, "message": r.message} for r in results],
        }

    @app.websocket("/ws/jobs/{job_id}")
    async def ws_job(websocket: WebSocket, job_id: str) -> None:
        await websocket.accept()
        app.state.sockets.add(websocket)
        try:
            try:
                uid = uuid.UUID(job_id)
            except ValueError:
                await websocket.send_json(
                    {"event": "error", "message": f"invalid job id: {job_id}"}
                )
                await websocket.close()
                return
            bus = app.state.buses.bus(str(uid))
            try:
                for event in bus.replay():
                    await websocket.send_json(event)
            except WebSocketDisconnect, RuntimeError:
                return  # gone before the live stream even started
            await _stream_live_events(bus, websocket, str(uid))
        except asyncio.CancelledError:
            # Task teardown (server shutdown, harness close, force quit):
            # the finally below already unwound everything (socket
            # discarded/closed, subscription closed), so there is nothing
            # left to do. Consume the cancellation so every supervisor
            # (uvicorn teardown, test harnesses) sees a normal exit
            # instead of a racy CancelledError. uncancel() keeps
            # asyncio's cancellation accounting honest (3.11+ protocol) —
            # this converts bookkeeping, it does not hide a hang: the
            # disconnect race above is what guarantees prompt exit.
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
        finally:
            app.state.sockets.discard(websocket)
            # Socket cleanup only: a close racing teardown may itself be
            # cancelled, which must not mask the real outcome.
            with contextlib.suppress(WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
                await websocket.close()

    static = Path(__file__).parent.parent / "ui" / "static"
    static.mkdir(parents=True, exist_ok=True)
    app.mount("/", StaticFiles(directory=str(static), html=True), name="ui")
    return app


__all__ = ["create_app", "problem"]
