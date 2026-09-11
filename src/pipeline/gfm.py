"""GFM lint + normalize + hard/warn classification.

Normalize first (mdformat-gfm), then lint (pymarkdownlnt, warnings only).
Hard failures come from the asset-integrity report: unresolved FIG
placeholders and missing/unreferenced assets fail the job (FR-QA-3, FR-IMG-6).
Lint rule hits are warnings — reported, non-fatal. Guardrail: the
normalizer must be idempotent and must not rewrite math/tables/emphasis;
test_normalize_idempotent pins that on fixtures.
"""

from dataclasses import dataclass, field

import mdformat


@dataclass
class LintVerdict:
    """QA gate input: hard fail blocks the job, warnings ride along."""

    hard_fail: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def normalize_markdown(markdown: str) -> str:
    """GFM normalization (syntax only — never source semantics)."""
    return mdformat.text(markdown, extensions={"gfm"})


def lint_markdown(markdown: str) -> list[str]:
    """Pymarkdownlnt rule hits as warning strings (empty when clean)."""
    from pymarkdown.api import PyMarkdownApi

    failures = PyMarkdownApi().scan_string(markdown).scan_failures
    return [
        f"{failure.rule_id}:{failure.line_number}:{failure.extra_error_information}"
        for failure in failures
    ]


def classify(
    integrity_ok: bool, integrity_reasons: list[str], lint_warnings: list[str]
) -> LintVerdict:
    """Hard-fail on integrity gaps; lint hits are warnings (FR-QA-3)."""
    if not integrity_ok:
        return LintVerdict(
            hard_fail=True, reasons=list(integrity_reasons), warnings=list(lint_warnings)
        )
    return LintVerdict(hard_fail=False, warnings=list(lint_warnings))


__all__ = ["LintVerdict", "classify", "lint_markdown", "normalize_markdown"]
