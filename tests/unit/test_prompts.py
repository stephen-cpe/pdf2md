"""versioned prompts, snapshot stability, version sensitivity."""

import hashlib

import pytest

from src.pipeline import agent as agent_mod
from src.pipeline.prompts import (
    PROMPT_VERSIONS,
    QA_TEMPLATE,
    TRANSCRIPTION_TEMPLATE,
    VERIFICATION_TEMPLATE,
    compute_pipeline_version,
)

# Snapshot hashes: changing a template REQUIRES updating these deliberately
# (prompt iteration) AND bumping that template's version identifier.
SNAPSHOTS = {
    "transcription": "1a93852306ffd420e4bfb41d811a6514696367541133f8907b21f9b00e762e5f",
    "verification": "ffce6fa4fffbce2304b63dc3991a7431c577f7107aaf9bf99b322aa884724bab",
    "qa": "2e9926df3d6c59ffde3f83cda3f50ddf4c4fa02f32d9e3f34217f0316be53fca",
}
TEMPLATES = {
    "transcription": TRANSCRIPTION_TEMPLATE,
    "verification": VERIFICATION_TEMPLATE,
    "qa": QA_TEMPLATE,
}


def test_versions_present_and_distinct() -> None:
    assert set(PROMPT_VERSIONS) == {"transcription", "verification", "qa"}
    assert len(set(PROMPT_VERSIONS.values())) == 3
    for version in PROMPT_VERSIONS.values():
        assert version.startswith(("transcription-v", "verification-v", "qa-v"))


def test_snapshots_stable() -> None:
    for name, template in TEMPLATES.items():
        digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
        assert digest == SNAPSHOTS[name], f"{name} template changed: bump its version + snapshot"


def test_agent_uses_library_not_own_copy() -> None:
    assert agent_mod.TRANSCRIPTION_SYSTEM == TRANSCRIPTION_TEMPLATE


@pytest.mark.parametrize(
    "clause",
    [
        "<<<MARKDOWN>>>",
        "ABSOLUTE PIXELS",
        "<!--FIG:page:",
        "MAY correct obvious source",
        'never emit the literal text "idx"',
    ],
)
def test_transcription_normative_clauses(clause: str) -> None:
    assert clause in TRANSCRIPTION_TEMPLATE


@pytest.mark.parametrize(
    "clause", ["<<<VERDICT>>>", "coverage", "FABRICATIONS", "TABLES", "FIGURES:"]
)
def test_verification_normative_clauses(clause: str) -> None:
    assert clause in VERIFICATION_TEMPLATE


@pytest.mark.parametrize("clause", ["<<<PATCHES>>>", "never be altered", "SUMMARY"])
def test_qa_normative_clauses(clause: str) -> None:
    assert clause in QA_TEMPLATE


def _base_kwargs() -> dict:
    return {
        "agent_model": "glm-5.3-flash",
        "ocr_model": "glm-ocr",
        "embed_model": "qwen3-embedding:0.6b",
        "render_dpi": 200,
        "rolling_context_pages": 3,
        "coverage_threshold": 95,
        "coverage_floor": 0.80,
        "hybrid_routing": False,
        "max_page_retries": 2,
        "thinking_transcribe": "low",
        "thinking_qa": "high",
        "toc_enabled": True,
        "fig_details": True,
    }


def test_pipeline_version_deterministic() -> None:
    assert compute_pipeline_version(**_base_kwargs()) == compute_pipeline_version(**_base_kwargs())


@pytest.mark.parametrize(
    "key",
    [
        "agent_model",
        "ocr_model",
        "embed_model",
        "render_dpi",
        "rolling_context_pages",
        "coverage_threshold",
        "coverage_floor",
        "hybrid_routing",
        "max_page_retries",
        "thinking_transcribe",
        "thinking_qa",
        "toc_enabled",
        "fig_details",
    ],
)
def test_pipeline_version_sensitive_to_every_input(key: str) -> None:
    base = compute_pipeline_version(**_base_kwargs())
    altered = dict(_base_kwargs())
    current = altered[key]
    altered[key] = not current if isinstance(current, bool) else "CHANGED"
    if isinstance(current, int) and not isinstance(current, bool):
        altered[key] = current + 1
    assert compute_pipeline_version(**altered) != base


def test_pipeline_version_sensitive_to_prompts() -> None:
    base = compute_pipeline_version(**_base_kwargs())
    other = dict(_base_kwargs())
    other_prompts = dict(PROMPT_VERSIONS)
    other_prompts["transcription"] = "transcription-v99-test"
    changed = compute_pipeline_version(prompts=other_prompts, **other)
    assert changed != base
