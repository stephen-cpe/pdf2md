"""Every Appendix B key typed + validated; secrets never in logs/repr.

Isolation: tests chdir to tmp_path so the real .env is never read and real
secrets never enter the test process; required secrets are fakes via env vars.
"""

import pytest
from pydantic import ValidationError

from src.config import Settings

FAKE_KEY = "test-fake-ollama-key"
FAKE_DB = "postgresql+asyncpg://pdf2md:test-fake-pw@localhost:5432/pdf2md"

REQUIRED_ENVS = {"OLLAMA_API_KEY": FAKE_KEY, "DATABASE_URL": FAKE_DB}


def _isolated(
    monkeypatch: pytest.MonkeyPatch, tmp_path, extra: dict[str, str] | None = None
) -> None:
    monkeypatch.chdir(tmp_path)
    for key, value in REQUIRED_ENVS.items():
        monkeypatch.setenv(key, value)
    for key, value in (extra or {}).items():
        monkeypatch.setenv(key, value)


def test_defaults_match_appendix_b(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _isolated(monkeypatch, tmp_path)
    s = Settings()  # type: ignore[call-arg]
    assert s.OLLAMA_LOCAL_URL == "http://localhost:11434"
    assert s.OLLAMA_CLOUD_URL == "https://ollama.com"
    assert s.AGENT_MODEL == "glm-5.3-flash"
    assert s.OCR_MODEL == "glm-ocr"
    assert s.AGENT_TIMEOUT_SECONDS == 300
    assert s.OCR_TIMEOUT_SECONDS == 120
    assert s.AGENT_MAX_OUTPUT_TOKENS is None
    assert s.KEEP_WORKSPACE_ON_SUCCESS is False
    assert s.MAX_CLEANUP_RETRIES == 3
    assert s.BIND_HOST == "127.0.0.1"
    assert s.PORT == 8000
    assert s.RENDER_DPI == 200
    assert s.ROLLING_CONTEXT_PAGES == 3
    assert s.COVERAGE_THRESHOLD == 95
    assert s.COVERAGE_FLOOR_TOKENS == 80
    assert s.COVERAGE_FLOOR_MIN_OCR_TOKENS == 30
    assert s.MAX_PAGE_RETRIES == 2
    assert s.MAX_PDF_MB == 500
    assert s.MAX_PDF_PAGES == 1000
    assert s.THINKING_EFFORT_TRANSCRIBE == "low"
    assert s.THINKING_EFFORT_DIAGRAM == "high"
    assert s.TOC_ENABLED is True
    assert s.FIG_DETAILS_BLOCKS is True
    # Diagram -> Mermaid primary capability.
    assert s.DIAGRAM_TO_MERMAID is True
    assert s.DIAGRAM_MIN_CONFIDENCE == 80
    assert s.DIAGRAM_VERIFY is True
    assert s.DIAGRAM_FALLBACK == "both"
    assert "flowchart" in s.allowed_diagram_types()
    assert "sequenceDiagram" in s.allowed_diagram_types()
    assert "xychart-beta" in s.allowed_diagram_types()
    # Reference routing + OCR switch.
    assert s.NATIVE_TEXT_FIRST is True
    assert s.NATIVE_TEXT_MIN_WORDS == 20
    assert s.OCR_ENABLED is True


def test_missing_api_key_fails_clear(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", FAKE_DB)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="OLLAMA_API_KEY"):
        Settings()  # type: ignore[call-arg]


def test_missing_database_url_fails_clear(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OLLAMA_API_KEY", FAKE_KEY)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValidationError, match="DATABASE_URL"):
        Settings()  # type: ignore[call-arg]


def test_secrets_never_in_repr_or_str(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _isolated(monkeypatch, tmp_path)
    s = Settings()  # type: ignore[call-arg]
    assert FAKE_KEY not in repr(s)
    assert FAKE_KEY not in str(s)
    assert "test-fake-pw" not in repr(s)
    assert "test-fake-pw" not in str(s)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("COVERAGE_THRESHOLD", "101"),
        ("COVERAGE_THRESHOLD", "-1"),
        ("COVERAGE_FLOOR_TOKENS", "101"),
        ("COVERAGE_FLOOR_TOKENS", "-1"),
        ("COVERAGE_FLOOR_MIN_OCR_TOKENS", "0"),
        ("RENDER_DPI", "0"),
        ("PORT", "0"),
        ("PORT", "99999"),
        ("MAX_PAGE_RETRIES", "-1"),
        ("AGENT_TIMEOUT_SECONDS", "0"),
        ("THINKING_EFFORT_TRANSCRIBE", "ultra"),
        ("THINKING_EFFORT_DIAGRAM", "medium"),
        ("DIAGRAM_MIN_CONFIDENCE", "101"),
        ("DIAGRAM_FALLBACK", "nonsense"),
    ],
)
def test_invalid_values_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path, key: str, value: str
) -> None:
    _isolated(monkeypatch, tmp_path, {key: value})
    with pytest.raises(ValidationError):
        Settings()  # type: ignore[call-arg]


def test_env_example_covers_all_keys_with_placeholders() -> None:
    from pathlib import Path

    example = Path(__file__).resolve().parents[2] / ".env.example"
    lines = example.read_text(encoding="utf-8").splitlines()
    active = {
        line.split("=", 1)[0].strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#") and "=" in line
    }
    mentioned = {
        line.lstrip("# ").split("=", 1)[0].strip() for line in lines if "=" in line and line.strip()
    }
    fields = set(Settings.model_fields)
    assert not (fields - mentioned), f".env.example undocumented keys: {fields - mentioned}"
    # Optional keys may ship commented-out; everything else must be active.
    assert not (fields - {"AGENT_MAX_OUTPUT_TOKENS"} - active), (
        f".env.example missing active keys: {fields - {'AGENT_MAX_OUTPUT_TOKENS'} - active}"
    )
    content = example.read_text(encoding="utf-8")
    assert "PASTE_YOUR_OLLAMA_CLOUD_KEY_HERE" in content
    assert "CHANGE_ME_APP_PASSWORD" in content
