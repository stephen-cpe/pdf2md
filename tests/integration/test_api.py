"""Contract tests: routes, 409 guard, problems, WS replay.

Integration tier: real DB via .env, mocked driver + health (no model calls).
Fake drivers are fast; job rows are uuid-isolated and cleaned up.
"""

import time
import uuid

import pymupdf
import pytest
from fastapi.testclient import TestClient

from src.api import app as app_mod
from src.api.app import create_app
from src.db import engine as engine_mod
from src.db import repo
from src.db.models import Job, JobStatus

pytestmark = pytest.mark.integration


def _pdf_bytes(pages: int = 1) -> bytes:
    doc = pymupdf.open()
    for number in range(pages):
        doc.new_page().insert_text((72, 72), f"contract page {number} " * 10)
    return doc.tobytes()


@pytest.fixture()
def client(monkeypatch):
    app = create_app()
    created: list[str] = []

    async def _fake_run_job(
        *, factory, settings, workspace, job_id, pdf_path, output_dir, options, control, emit
    ):
        from src.db.models import JobStatus as JS

        async with factory() as session:
            await repo.set_job_status(session, job_id, JS.COMPLETED, "done")
        await emit({"event": "job_finished", "job_id": str(job_id), "status": "completed"})
        return "completed"

    def _fake_health(settings):
        from src.health import HealthResult

        return [HealthResult("ollama_local", True, "fake ok")]

    monkeypatch.setattr(app_mod.driver_mod, "run_job", _fake_run_job)
    monkeypatch.setattr(app_mod.health_mod, "run_all", _fake_health)
    with TestClient(app) as test_client:
        yield test_client, app, created
    # Cleanup job rows left by the flow.
    import asyncio

    async def _cleanup():
        dsn = app.state.settings.DATABASE_URL.get_secret_value()
        engine = engine_mod.create_engine(dsn)
        factory = engine_mod.session_factory(engine)
        try:
            async with factory() as session:
                for job_id in created:
                    job = await session.get(Job, uuid.UUID(job_id))
                    if job is not None:
                        await session.delete(job)
                await session.commit()
        finally:
            await engine.dispose()

    asyncio.run(_cleanup())


def _post(client, tmp_path, name="t.pdf", content=None, options="{}"):
    return client.post(
        "/api/v1/jobs",
        files={"file": (name, content or _pdf_bytes(), "application/pdf")},
        data={"output_dir": str(tmp_path / "out"), "options": options},
    )


def _wait_status(client, job_id: str, want: str, tries: int = 100) -> dict:
    for _ in range(tries):
        detail = client.get(f"/api/v1/jobs/{job_id}").json()
        if detail["status"] == want:
            return detail
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {want}: {detail}")


def test_post_get_flow(client, tmp_path) -> None:
    test_client, _app, created = client
    resp = _post(test_client, tmp_path)
    assert resp.status_code == 201
    job_id = resp.json()["job_id"]
    created.append(job_id)
    detail = _wait_status(test_client, job_id, "completed")
    assert detail["filename"] == "t.pdf" and detail["pages_done"] == 0
    assert detail["live"] is False
    assert test_client.get(f"/api/v1/jobs/{job_id}/pages").json() == []
    assert test_client.get(f"/api/v1/jobs/{job_id}/preview").json()["markdown"] == ""
    report = test_client.get(f"/api/v1/jobs/{job_id}/report")
    assert report.status_code == 200 and "Conversion report" in report.json()["report"]
    events = test_client.get(f"/api/v1/jobs/{job_id}/events").json()
    assert isinstance(events, list)
    listing = test_client.get("/api/v1/jobs").json()
    assert any(entry["job_id"] == job_id for entry in listing)


def test_409_while_running(client, tmp_path, monkeypatch) -> None:
    import asyncio as _asyncio

    test_client, _app, created = client
    release = _asyncio.Event()

    async def _slow(*, control, emit, job_id, **kwargs):
        await release.wait()

    monkeypatch.setattr(app_mod.driver_mod, "run_job", _slow)
    first = _post(test_client, tmp_path)
    assert first.status_code == 201
    created.append(first.json()["job_id"])
    assert test_client.get(f"/api/v1/jobs/{first.json()['job_id']}").json()["live"] is True
    second = _post(test_client, tmp_path)
    assert second.status_code == 409
    body = second.json()
    assert body["title"] == "job running" and second.headers["content-type"].startswith(
        "application/problem+json"
    )
    release.set()


def test_cancel_pause_resume(client, tmp_path, monkeypatch) -> None:
    import asyncio as _asyncio

    test_client, _app, created = client

    async def _waiter(*, factory, job_id, control, emit, **kwargs):
        for _ in range(200):
            if control.cancel_requested:
                async with factory() as session:
                    await repo.set_job_status(session, job_id, JobStatus.CANCELLED, "cancelled")
                return "cancelled"
            await _asyncio.sleep(0.02)
        return "completed"

    monkeypatch.setattr(app_mod.driver_mod, "run_job", _waiter)
    job_id = _post(test_client, tmp_path).json()["job_id"]
    created.append(job_id)
    assert test_client.post(f"/api/v1/jobs/{job_id}/pause").status_code == 200
    assert test_client.post(f"/api/v1/jobs/{job_id}/resume").status_code == 200
    assert test_client.post(f"/api/v1/jobs/{job_id}/cancel").status_code == 200
    _wait_status(test_client, job_id, "cancelled")


def test_problem_shapes(client, tmp_path) -> None:
    test_client, _app, _created = client
    missing = test_client.get("/api/v1/jobs/00000000-0000-0000-0000-000000000000")
    assert missing.status_code == 404
    assert missing.json()["title"] and missing.json()["status"] == 404
    bad_id = test_client.get("/api/v1/jobs/not-a-uuid")
    assert bad_id.status_code == 422
    not_pdf = test_client.post(
        "/api/v1/jobs",
        files={"file": ("t.txt", b"hello", "text/plain")},
        data={"output_dir": str(tmp_path), "options": "{}"},
    )
    assert not_pdf.status_code == 422
    bad_opts = _post(test_client, tmp_path, options="{oops")
    assert bad_opts.status_code == 422
    report = test_client.get("/api/v1/jobs/00000000-0000-0000-0000-000000000000/report")
    assert report.status_code == 404


def test_health_uses_service(client) -> None:
    test_client, _app, _created = client
    body = test_client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["components"][0]["name"] == "ollama_local"


def test_browse_drives_and_dirs(client, tmp_path) -> None:
    test_client, _app, _created = client
    roots = test_client.get("/api/v1/browse").json()
    assert roots["parent"] is None and len(roots["entries"]) >= 1
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    (tmp_path / "note.txt").write_text("x")
    body = test_client.get("/api/v1/browse", params={"path": str(tmp_path)}).json()
    assert [e["name"] for e in body["entries"]] == ["alpha", "beta"]
    assert body["parent"] == str(tmp_path.parent)
    assert test_client.get("/api/v1/browse", params={"path": "relative"}).status_code == 422
    assert (
        test_client.get("/api/v1/browse", params={"path": str(tmp_path / "nope")}).status_code
        == 404
    )
    assert (
        test_client.get("/api/v1/browse", params={"path": str(tmp_path / "note.txt")}).status_code
        == 404
    )


def test_upload_preserves_filename(client, tmp_path) -> None:
    import shutil
    from pathlib import Path

    test_client, _app, created = client
    resp = _post(test_client, tmp_path, name="My Report 2024.pdf")
    assert resp.status_code == 201
    job_id = resp.json()["job_id"]
    created.append(job_id)
    detail = test_client.get(f"/api/v1/jobs/{job_id}").json()
    assert detail["filename"] == "My Report 2024.pdf"
    workspace_file = Path("workspace") / "jobs" / job_id / "My_Report_2024.pdf"
    assert workspace_file.is_file()
    shutil.rmtree(Path("workspace") / "jobs" / job_id, ignore_errors=True)


def test_restart_paths(client, tmp_path) -> None:
    import asyncio as _asyncio
    import shutil
    from pathlib import Path as _Path

    from src.db.models import JobStatus as _JobStatus

    test_client, app, created = client
    settings = app.state.settings
    workspace = app.state.workspace

    async def _paused_id() -> str:
        dsn = settings.DATABASE_URL.get_secret_value()
        engine = engine_mod.create_engine(dsn)
        factory = engine_mod.session_factory(engine)
        try:
            async with factory() as session:
                job = await repo.create_job(
                    session,
                    filename="restart-me.pdf",
                    file_sha256="k" * 64,
                    page_count=1,
                    output_dir=str(tmp_path),
                    options={},
                )
                await repo.set_job_status(session, job.id, _JobStatus.PAUSED, "x")
                return str(job.id)
        finally:
            await engine.dispose()

    paused_id = _asyncio.run(_paused_id())
    created.append(paused_id)
    # No workspace PDF yet → 409.
    assert test_client.post(f"/api/v1/jobs/{paused_id}/restart").status_code == 409
    workspace.job_dir(paused_id).mkdir(parents=True, exist_ok=True)
    (workspace.job_dir(paused_id) / "restart-me.pdf").write_bytes(b"%PDF-1.4 fake")
    resp = test_client.post(f"/api/v1/jobs/{paused_id}/restart")
    assert resp.status_code == 202
    assert _wait_status(test_client, paused_id, "completed")["status"] == "completed"
    shutil.rmtree(_Path("workspace") / "jobs" / paused_id, ignore_errors=True)


def test_control_409_explains_state(client, tmp_path) -> None:
    """No live control: 409 names the DB state + the way back (incident 5fbf)."""
    import asyncio as _asyncio

    test_client, app, created = client

    async def _row(status) -> str:
        dsn = app.state.settings.DATABASE_URL.get_secret_value()
        engine = engine_mod.create_engine(dsn)
        factory = engine_mod.session_factory(engine)
        try:
            async with factory() as session:
                job = await repo.create_job(
                    session,
                    filename="stalled.pdf",
                    file_sha256="n" * 64,
                    page_count=1,
                    output_dir=str(tmp_path),
                    options={},
                )
                await repo.set_job_status(session, job.id, status, "x")
                return str(job.id)
        finally:
            await engine.dispose()

    paused_id = _asyncio.run(_row(JobStatus.PAUSED))
    created.append(paused_id)
    resp = test_client.post(f"/api/v1/jobs/{paused_id}/pause")
    assert resp.status_code == 409
    assert "restart" in resp.json()["detail"]

    done_id = _asyncio.run(_row(JobStatus.COMPLETED))
    created.append(done_id)
    resp = test_client.post(f"/api/v1/jobs/{done_id}/cancel")
    assert resp.status_code == 409
    assert "already completed" in resp.json()["detail"]


def test_artifact_fallback_without_recorded_paths(client, tmp_path) -> None:
    """Jobs completed before artifacts were recorded still download via layout."""
    import asyncio as _asyncio

    test_client, app, created = client

    async def _row() -> str:
        dsn = app.state.settings.DATABASE_URL.get_secret_value()
        engine = engine_mod.create_engine(dsn)
        factory = engine_mod.session_factory(engine)
        try:
            async with factory() as session:
                job = await repo.create_job(
                    session,
                    filename="Flat Doc.pdf",
                    file_sha256="m" * 64,
                    page_count=1,
                    output_dir=str(tmp_path),
                    options={},
                )
                await repo.set_job_status(session, job.id, JobStatus.COMPLETED, "done")
                return str(job.id)
        finally:
            await engine.dispose()

    job_id = _asyncio.run(_row())
    created.append(job_id)
    (tmp_path / "Flat_Doc.md").write_text("# flat", encoding="utf-8")
    (tmp_path / "conversion-report.md").write_text("# report", encoding="utf-8")
    doc = test_client.get(f"/api/v1/jobs/{job_id}/artifact/document")
    assert doc.status_code == 200 and "# flat" in doc.text
    report = test_client.get(f"/api/v1/jobs/{job_id}/artifact/report")
    assert report.status_code == 200
    assert test_client.get(f"/api/v1/jobs/{job_id}/artifact/other").status_code == 404


def test_artifact_fallback_prefers_per_doc_subfolder(client, tmp_path) -> None:
    """Top-level .md + per-doc assets/report; nested .md as fallback."""
    import asyncio as _asyncio

    test_client, app, created = client

    async def _row() -> str:
        dsn = app.state.settings.DATABASE_URL.get_secret_value()
        engine = engine_mod.create_engine(dsn)
        factory = engine_mod.session_factory(engine)
        try:
            async with factory() as session:
                job = await repo.create_job(
                    session,
                    filename="My Doc.pdf",
                    file_sha256="n" * 64,
                    page_count=1,
                    output_dir=str(tmp_path),
                    options={},
                )
                await repo.set_job_status(session, job.id, JobStatus.COMPLETED, "done")
                return str(job.id)
        finally:
            await engine.dispose()

    job_id = _asyncio.run(_row())
    created.append(job_id)
    # Current layout: top-level .md wins; report lives in the per-doc folder.
    (tmp_path / "My_Doc").mkdir()
    (tmp_path / "My_Doc.md").write_text("# new top-level", encoding="utf-8")
    (tmp_path / "My_Doc" / "My_Doc.md").write_text("# nested copy", encoding="utf-8")
    (tmp_path / "My_Doc" / "conversion-report.md").write_text("# new report", encoding="utf-8")
    (tmp_path / "conversion-report.md").write_text("# stale flat report", encoding="utf-8")
    doc = test_client.get(f"/api/v1/jobs/{job_id}/artifact/document")
    assert doc.status_code == 200 and "# new top-level" in doc.text
    report = test_client.get(f"/api/v1/jobs/{job_id}/artifact/report")
    assert report.status_code == 200 and "# new report" in report.text


def test_ws_ordered_and_replay(client) -> None:
    test_client, app, _created = client
    job_id = str(uuid.uuid4())
    bus = app.state.buses.bus(job_id)
    events = [{"event": "log", "job_id": job_id, "message": f"m{n}"} for n in range(3)]

    async def _publish():
        for event in events:
            await bus.publish(event)

    import asyncio as _asyncio

    _asyncio.run(_publish())
    with test_client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        received = [ws.receive_json() for _ in range(3)]
    assert [e["message"] for e in received] == ["m0", "m1", "m2"]

    async def _publish_one():
        await bus.publish({"event": "log", "job_id": job_id, "message": "m3"})

    _asyncio.run(_publish_one())
    with test_client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        replayed = [ws.receive_json() for _ in range(4)]
    assert [e["message"] for e in replayed] == ["m0", "m1", "m2", "m3"]


def _ws_endpoint(app):
    for route in app.routes:
        if getattr(route, "path", "") == "/ws/jobs/{job_id}":
            return route.endpoint
    raise AssertionError("ws route missing")


async def test_ws_disconnect_exits_and_cancel_cleans_up(client) -> None:
    """Disconnect ends the stream (no hang); teardown cancels exit quietly."""
    import asyncio as _asyncio

    from starlette.websockets import WebSocketDisconnect as _WSD

    _test_client, app, _created = client
    endpoint = _ws_endpoint(app)

    class _DisconnectingSocket:
        """receive() reports death at once — the parked-subscribe hang would
        ignore this forever, so guard the call with a timeout."""

        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def accept(self) -> None:
            return None

        async def send_json(self, event: dict) -> None:
            self.sent.append(event)

        async def receive(self) -> dict:
            raise _WSD(code=1001)

        async def close(self, code: int = 1000) -> None:
            return None

    # 1. Disconnect with live events flowing: returns promptly, socket dropped.
    live_id = str(uuid.uuid4())
    await app.state.buses.bus(live_id).publish({"event": "log", "job_id": live_id, "message": "m0"})
    dying = _DisconnectingSocket()
    await _asyncio.wait_for(endpoint(dying, live_id), timeout=5.0)
    assert dying.sent and dying.sent[0]["message"] == "m0"  # replay flushed first
    assert dying not in app.state.sockets

    # 2. Task cancelled while parked: CancelledError propagates (correct
    # asyncio citizenship — teardown retrieves it quietly), socket dropped,
    # subscriber removed.
    class _ParkingSocket:
        async def accept(self) -> None:
            return None

        async def send_json(self, event: dict) -> None:
            return None

        async def receive(self) -> dict:
            await _asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def close(self, code: int = 1000) -> None:
            return None

    wait_id = str(uuid.uuid4())
    parking = _ParkingSocket()
    task = _asyncio.create_task(endpoint(parking, wait_id))
    await _asyncio.sleep(0.1)
    task.cancel()
    await task  # consumed via uncancel(): normal exit, nothing stranded
    assert parking not in app.state.sockets
    assert app.state.buses.bus(wait_id)._subscribers == set()

    # 3. Close racing teardown may itself be cancelled: still quiet.
    class _CloseDiesSocket(_ParkingSocket):
        async def close(self, code: int = 1000) -> None:
            raise _asyncio.CancelledError()

    close_id = str(uuid.uuid4())
    close_dies = _CloseDiesSocket()
    task2 = _asyncio.create_task(endpoint(close_dies, close_id))
    await _asyncio.sleep(0.1)
    task2.cancel()
    await task2  # close racing teardown stays quiet too
    assert close_dies not in app.state.sockets
