"""versioned prompts, snapshot stability, version sensitivity."""

import hashlib

import pytest

from src.pipeline import agent as agent_mod
from src.pipeline.prompts import (
    DIAGRAM_TEMPLATE,
    DIAGRAM_VERIFY_TEMPLATE,
    PROMPT_VERSIONS,
    TRANSCRIPTION_TEMPLATE,
    VERIFICATION_TEMPLATE,
    compute_pipeline_version,
)

# Snapshot hashes: changing a template REQUIRES updating these deliberately
# (prompt iteration) AND bumping that template's version identifier.
SNAPSHOTS = {
    "transcription": "1a93852306ffd420e4bfb41d811a6514696367541133f8907b21f9b00e762e5f",
    "verification": "ffce6fa4fffbce2304b63dc3991a7431c577f7107aaf9bf99b322aa884724bab",
    "diagram": "ca20568644ff8c6affec03652ea03edd5c281ea8974064289c16615e3192f324",
    "diagram_verification": "fa2608014b2ebdae4c8a321b60462e1370a8b1b386612a774815d0042749065c",
}
TEMPLATES = {
    "transcription": TRANSCRIPTION_TEMPLATE,
    "verification": VERIFICATION_TEMPLATE,
    "diagram": DIAGRAM_TEMPLATE,
    "diagram_verification": DIAGRAM_VERIFY_TEMPLATE,
}


def test_versions_present_and_distinct() -> None:
    assert set(PROMPT_VERSIONS) == {
        "transcription",
        "verification",
        "diagram",
        "diagram_verification",
    }
    assert len(set(PROMPT_VERSIONS.values())) == 4
    for version in PROMPT_VERSIONS.values():
        assert version.startswith(
            ("transcription-v", "verification-v", "diagram-v", "diagram-verification-v")
        )


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


@pytest.mark.parametrize(
    "clause", ["<<<DIAGRAM>>>", "<<<MERMAID>>>", "<<<DATA>>>", "convertible", "NEVER invent"]
)
def test_diagram_normative_clauses(clause: str) -> None:
    assert clause in DIAGRAM_TEMPLATE


@pytest.mark.parametrize("clause", ["<<<VERDICT>>>", "FABRICATIONS", "coverage"])
def test_diagram_verification_normative_clauses(clause: str) -> None:
    assert clause in DIAGRAM_VERIFY_TEMPLATE


def _base_kwargs() -> dict:
    return {
        "agent_model": "glm-5.3-flash",
        "ocr_model": "glm-ocr",
        "render_dpi": 200,
        "rolling_context_pages": 3,
        "coverage_threshold": 95,
        "coverage_floor": 0.80,
        "max_page_retries": 2,
        "thinking_transcribe": "low",
        "thinking_diagram": "high",
        "native_text_first": True,
        "native_text_min_words": 20,
        "toc_enabled": True,
        "fig_details": True,
        "diagram_to_mermaid": True,
        "diagram_min_confidence": 80,
        "diagram_verify": True,
        "diagram_fallback": "both",
        "diagram_keep_image": True,
    }


def test_pipeline_version_deterministic() -> None:
    assert compute_pipeline_version(**_base_kwargs()) == compute_pipeline_version(**_base_kwargs())


@pytest.mark.parametrize(
    "key",
    [
        "agent_model",
        "ocr_model",
        "render_dpi",
        "rolling_context_pages",
        "coverage_threshold",
        "coverage_floor",
        "max_page_retries",
        "thinking_transcribe",
        "thinking_diagram",
        "native_text_first",
        "native_text_min_words",
        "toc_enabled",
        "fig_details",
        "diagram_to_mermaid",
        "diagram_min_confidence",
        "diagram_verify",
        "diagram_fallback",
        "diagram_keep_image",
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


def test_pipeline_version_covers_every_job_option() -> None:
    """Drift guard: every JobOptions field must be a fingerprint input.

    A behavior-changing option absent from compute_pipeline_version would let
    a resumed job silently mix pipeline behavior.
    """
    import dataclasses

    from src.pipeline.driver import JobOptions

    job_option_fields = {f.name for f in dataclasses.fields(JobOptions)}
    fingerprint_inputs = set(_base_kwargs())
    missing = job_option_fields - fingerprint_inputs
    assert not missing, f"JobOptions fields missing from pipeline_version: {sorted(missing)}"
