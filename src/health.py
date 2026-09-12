"""Component health checks.

Each check returns ok/fail + an actionable message (FR-UI-6, §5.1 GET /health
contract foundation — the REST endpoint reuses run_all).

Test seams: the raw helpers isolate all I/O (HTTP, Postgres) so unit tests
mock them without network/DB. Messages NEVER contain secrets — Postgres
errors show host/db only, never the DSN password.
"""

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from src.config import Settings

CLOUD_SUFFIX = ":cloud"
DEFAULT_MIN_FREE_BYTES = 500 * 1024 * 1024


@dataclass(frozen=True)
class HealthResult:
    """One component's verdict."""

    name: str
    ok: bool
    message: str


# --- Raw I/O seams (mocked in unit tests) ---


def _http_get_json(url: str, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    """GET JSON via httpx. Raises httpx.HTTPError / ValueError on bad JSON."""
    with httpx.Client(timeout=timeout) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data


def _pg_select_1(dsn: str, timeout: float) -> None:
    """Connect + SELECT 1 via asyncpg. Raises on any failure."""
    import asyncpg  # type: ignore[import-untyped]

    async def _ping() -> None:
        conn = await asyncpg.connect(dsn, timeout=timeout)
        try:
            await conn.fetchval("SELECT 1")
        finally:
            await conn.close()

    asyncio.run(_ping())


# --- Checks ---


def check_ollama_local(base_url: str, timeout: float = 10.0) -> HealthResult:
    """Local daemon reachable? (No auth.)"""
    try:
        _http_get_json(f"{base_url}/api/version", {}, timeout)
    except Exception as exc:
        return HealthResult(
            "ollama_local",
            False,
            f"Ollama local unreachable at {base_url}: {exc}. "
            "Start Ollama from the Start menu and retry.",
        )
    return HealthResult("ollama_local", True, f"Ollama local responding at {base_url}.")


def _canonical(name: str) -> str:
    """Bare `model` means `model:latest` (Ollama CLI convention)."""
    return name if ":" in name else f"{name}:latest"


def _model_names(tags: dict[str, Any]) -> list[str]:
    return [str(m.get("name", "")) for m in tags.get("models", [])]


def _resolved(wanted: str, names: list[str]) -> bool:
    """Exact or tag-defaulted match: `glm-ocr` resolves `glm-ocr:latest`."""
    canon = {_canonical(n) for n in names}
    return wanted in names or _canonical(wanted) in canon


def check_local_models(base_url: str, ocr_model: str, timeout: float = 10.0) -> HealthResult:
    """Required local models pulled?"""
    try:
        names = _model_names(_http_get_json(f"{base_url}/api/tags", {}, timeout))
    except Exception as exc:
        return HealthResult(
            "local_models",
            False,
            f"Could not list local models at {base_url}: {exc}. Is the Ollama daemon running?",
        )
    missing = [m for m in (ocr_model,) if not _resolved(m, names)]
    if missing:
        pulls = "  ".join(f"ollama pull {m}" for m in missing)
        return HealthResult(
            "local_models", False, f"Local model(s) missing: {missing}. Run: {pulls}"
        )
    return HealthResult("local_models", True, f"Local models present: {ocr_model}.")


def normalize_agent_model(model: str) -> str:
    """Both name forms (§5.3) resolve to the Cloud listing's bare name."""
    return model.removesuffix(CLOUD_SUFFIX)


def check_ollama_cloud(
    cloud_url: str, api_key: str, agent_model: str, timeout: float = 15.0
) -> HealthResult:
    """Cloud key valid + AGENT_MODEL resolvable via preflight /api/tags."""
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        names = _model_names(_http_get_json(f"{cloud_url}/api/tags", headers, timeout))
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (401, 403):
            return HealthResult(
                "ollama_cloud",
                False,
                "Ollama Cloud rejected the API key (HTTP "
                f"{exc.response.status_code}). Check OLLAMA_API_KEY in .env.",
            )
        return HealthResult(
            "ollama_cloud",
            False,
            f"Ollama Cloud error (HTTP {exc.response.status_code}): "
            "retry later; sustained outage pauses jobs (FR-AGT-8).",
        )
    except Exception as exc:
        return HealthResult(
            "ollama_cloud",
            False,
            f"Ollama Cloud unreachable at {cloud_url}: {exc}. Check network connection.",
        )
    wanted = normalize_agent_model(agent_model)
    if not _resolved(wanted, names) and not _resolved(agent_model, names):
        return HealthResult(
            "ollama_cloud",
            False,
            f"AGENT_MODEL '{agent_model}' not in Cloud model list. "
            "Check the name and subscription.",
        )
    return HealthResult(
        "ollama_cloud", True, f"Cloud key valid; AGENT_MODEL '{agent_model}' resolves."
    )


def _safe_pg_label(dsn: str) -> str:
    """host/db for messages — password never leaves the SecretStr."""
    parsed = urlparse(dsn.replace("+asyncpg", ""))
    return f"{parsed.hostname}/{parsed.path.lstrip('/')}"


def check_postgres(dsn: str, timeout: float = 10.0) -> HealthResult:
    """Postgres reachable + auth OK?"""
    label = _safe_pg_label(dsn)
    try:
        _pg_select_1(dsn.replace("+asyncpg", ""), timeout)
    except Exception as exc:
        return HealthResult(
            "postgres",
            False,
            f"PostgreSQL unreachable at {label}: {exc}. "
            "Verify the service is running and DATABASE_URL in .env.",
        )
    return HealthResult("postgres", True, f"PostgreSQL responding at {label}.")


def check_disk(path: str, min_free_bytes: int = DEFAULT_MIN_FREE_BYTES) -> HealthResult:
    """Enough free space where outputs/workspaces land?"""
    target = Path(path).resolve()
    target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target).free
    if free < min_free_bytes:
        return HealthResult(
            "disk",
            False,
            f"Only {free // 1024 // 1024} MB free at {target} "
            f"(need {min_free_bytes // 1024 // 1024} MB). Free up disk space.",
        )
    return HealthResult("disk", True, f"{free // 1024 // 1024} MB free at {target}.")


def run_all(settings: Settings) -> list[HealthResult]:
    """Run every check; order matches the handoff checklist."""
    key = settings.OLLAMA_API_KEY.get_secret_value()
    db = settings.DATABASE_URL.get_secret_value()
    return [
        check_ollama_local(settings.OLLAMA_LOCAL_URL),
        check_local_models(settings.OLLAMA_LOCAL_URL, settings.OCR_MODEL),
        check_ollama_cloud(settings.OLLAMA_CLOUD_URL, key, settings.AGENT_MODEL),
        check_postgres(db),
        check_disk("."),
    ]


__all__ = [
    "HealthResult",
    "check_disk",
    "check_local_models",
    "check_ollama_cloud",
    "check_ollama_local",
    "check_postgres",
    "normalize_agent_model",
    "run_all",
]
