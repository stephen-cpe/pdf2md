"""Unit-tested with mocked failures (no network/DB/Ollama)."""

from collections import namedtuple

import httpx
import pytest

from src import health
from src.health import (
    HealthResult,
    check_chroma,
    check_disk,
    check_local_models,
    check_ollama_cloud,
    check_ollama_local,
    check_postgres,
    normalize_agent_model,
    run_all,
)

_DiskUsage = namedtuple("_DiskUsage", ["total", "used", "free"])
GB = 1024 * 1024 * 1024


def _http_error(status: int) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "https://ollama.com/api/tags")
    return httpx.HTTPStatusError("err", request=req, response=httpx.Response(status, request=req))


# --- ollama local ---


def test_local_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "_http_get_json", lambda *a, **k: {"version": "0.1"})
    result = check_ollama_local("http://localhost:11434")
    assert result == HealthResult("ollama_local", True, result.message)


def test_local_down_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(health, "_http_get_json", _boom)
    result = check_ollama_local("http://localhost:11434")
    assert not result.ok
    assert "Start Ollama" in result.message


# --- local models ---


def test_local_models_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        health, "_http_get_json", lambda *a, **k: {"models": [{"name": "glm-ocr"}, {"name": "q"}]}
    )
    assert check_local_models("http://x", "glm-ocr", "q").ok


def test_local_models_bare_name_matches_latest_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        health,
        "_http_get_json",
        lambda *a, **k: {"models": [{"name": "glm-ocr:latest"}, {"name": "qwen3-embedding:0.6b"}]},
    )
    assert check_local_models("http://x", "glm-ocr", "qwen3-embedding:0.6b").ok


def test_local_models_missing_names_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "_http_get_json", lambda *a, **k: {"models": []})
    result = check_local_models("http://x", "glm-ocr", "q")
    assert not result.ok
    assert "ollama pull glm-ocr" in result.message


def test_local_models_tags_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(health, "_http_get_json", _boom)
    assert not check_local_models("http://x", "glm-ocr", "q").ok


# --- cloud ---


def test_normalize_both_name_forms() -> None:
    assert normalize_agent_model("glm-5.3-flash") == "glm-5.3-flash"
    assert normalize_agent_model("glm-5.3-flash:cloud") == "glm-5.3-flash"


def _tags(names: list[str]):
    return lambda *a, **k: {"models": [{"name": n} for n in names]}


def test_cloud_ok_both_forms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "_http_get_json", _tags(["glm-5.3-flash"]))
    assert check_ollama_cloud("https://ollama.com", "k", "glm-5.3-flash").ok
    assert check_ollama_cloud("https://ollama.com", "k", "glm-5.3-flash:cloud").ok


def test_cloud_401_points_at_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a, **k):
        raise _http_error(401)

    monkeypatch.setattr(health, "_http_get_json", _boom)
    result = check_ollama_cloud("https://ollama.com", "bad", "glm-5.3-flash")
    assert not result.ok
    assert "OLLAMA_API_KEY" in result.message
    assert "bad" not in result.message


def test_cloud_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a, **k):
        raise httpx.ConnectError("dns")

    monkeypatch.setattr(health, "_http_get_json", _boom)
    assert not check_ollama_cloud("https://ollama.com", "k", "glm-5.3-flash").ok


def test_cloud_model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "_http_get_json", _tags(["other-model"]))
    result = check_ollama_cloud("https://ollama.com", "k", "glm-5.3-flash")
    assert not result.ok
    assert "glm-5.3-flash" in result.message


# --- postgres ---


def test_postgres_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "_pg_select_1", lambda *a, **k: None)
    dsn = "postgresql+asyncpg://pdf2md:secret@localhost:5432/pdf2md"
    result = check_postgres(dsn)
    assert result.ok
    assert "secret" not in result.message


def test_postgres_fail_hides_password(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a, **k):
        raise ConnectionRefusedError("nope")

    monkeypatch.setattr(health, "_pg_select_1", _boom)
    result = check_postgres("postgresql+asyncpg://pdf2md:s3cr3t@localhost:5432/pdf2md")
    assert not result.ok
    assert "s3cr3t" not in result.message
    assert "localhost" in result.message


# --- chroma / disk ---


def test_chroma_ok_and_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "_chroma_heartbeat", lambda *a, **k: 1)
    assert check_chroma("./chroma").ok

    def _boom(*a, **k):
        raise RuntimeError("locked")

    monkeypatch.setattr(health, "_chroma_heartbeat", _boom)
    assert not check_chroma("./chroma").ok


def test_disk_ok_and_full(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda p: _DiskUsage(10 * GB, 1 * GB, 9 * GB))
    assert check_disk(str(tmp_path)).ok
    monkeypatch.setattr(shutil, "disk_usage", lambda p: _DiskUsage(10 * GB, 9 * GB, 1 * GB - 1))
    result = check_disk(str(tmp_path), min_free_bytes=GB)
    assert not result.ok


# --- composition ---


def test_run_all_composes(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import shutil

    monkeypatch.setattr(
        health, "_http_get_json", _tags(["glm-ocr", "qwen3-embedding:0.6b", "glm-5.3-flash"])
    )
    monkeypatch.setattr(health, "_pg_select_1", lambda *a, **k: None)
    monkeypatch.setattr(health, "_chroma_heartbeat", lambda *a, **k: 1)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: _DiskUsage(10 * GB, 1 * GB, 9 * GB))

    from src.config import Settings

    for key in ("OLLAMA_API_KEY", "DATABASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OLLAMA_API_KEY", "k")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/db")
    results = run_all(Settings())  # type: ignore[call-arg]
    assert [r.name for r in results] == [
        "ollama_local",
        "local_models",
        "ollama_cloud",
        "postgres",
        "chroma",
        "disk",
    ]
    assert all(r.ok for r in results)


def test_degraded_when_postgres_down(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """§8 degraded mode: PG down degrades health, never crashes it, leaks nothing."""
    import shutil

    monkeypatch.setattr(
        health, "_http_get_json", _tags(["glm-ocr", "qwen3-embedding:0.6b", "glm-5.3-flash"])
    )

    def _pg_down(*a, **k):
        raise ConnectionRefusedError("pg down")

    monkeypatch.setattr(health, "_pg_select_1", _pg_down)
    monkeypatch.setattr(health, "_chroma_heartbeat", lambda *a, **k: 1)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: _DiskUsage(10 * GB, 1 * GB, 9 * GB))

    from src.config import Settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OLLAMA_API_KEY", "k")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:s3cr3t@localhost:5432/db")
    results = {r.name: r for r in run_all(Settings())}  # type: ignore[call-arg]
    assert results["postgres"].ok is False
    assert "s3cr3t" not in results["postgres"].message
    assert results["ollama_local"].ok and results["ollama_cloud"].ok
    assert not all(r.ok for r in results.values())  # overall: degraded
