"""GFM lint + normalize + hard/warn classification + GitHub math safety.

Normalize first (mdformat-gfm), then lint (pymarkdownlnt, warnings only).
Hard failures come from the asset-integrity report: unresolved FIG
placeholders and missing/unreferenced assets fail the job (FR-QA-3, FR-IMG-6).
Lint rule hits are warnings — reported, non-fatal. Guardrail: the
normalizer must be idempotent and must not rewrite math/tables/emphasis;
test_gfm.py pins that on fixtures.

`github_math_safe()` is an output-time transform (applied by
report.write_output): GitHub's Markdown pipeline strips one level of
backslash-escaping from `$...$` before KaTeX renders it, so `\\_` inside
`\\text{}` reaches KaTeX as a bare `_` ("'_' allowed only in math mode") and
`\\{`/`\\}` lose their braces. Double-escaping is NOT the fix — GitHub then
drops the whole formula out of math mode and renders it as broken emphasis.
Validated against `POST https://api.github.com/markdown` + KaTeX: drawing the
underscore with `\\rule` and using `\\lbrace`/`\\rbrace` keeps the formula
recognized as math and renders cleanly on both sides.
"""

import re
from dataclasses import dataclass, field

import mdformat

# mdformat has no math extension here, so it treats `$...$`/`$$...$$` content
# as plain text and escapes it: `\text{old\_way}` becomes `\\text{old_way}` —
# invalid LaTeX (a `\\` line break plus a misplaced `_`), which GitHub then
# rejects with errors like "'_' allowed only in math mode". Math spans are
# stashed verbatim across the mdformat pass and restored untouched.
_MATH_SPAN = re.compile(r"\$\$.*?\$\$|\$[^$\n]+?\$", re.DOTALL)
_MATH_STASH = "@@PDF2MD-MATH-{}@@"
_MATH_RESTORE = re.compile(r"@@PDF2MD-MATH-(\d+)@@")

# GitHub-math-safe replacements (validated against GitHub's Markdown API).
_UNDERSCORE_IN_TEXT = re.compile(r"\\text\{([^{}]*?)\\_([^{}]*?)\}")
_TEXT_UNDERSCORE_RULE = r"\rule[-0.05em]{0.35em}{0.5pt}"
# Braces: `\lbrace`/`\rbrace` need a separating space or the following/
# preceding letter merges into an undefined macro (`\lbracea`). Math mode
# collapses whitespace, so the padding is invisible in the render.
_OPEN_BRACE = re.compile(r"\\\{(?!\s)")
_CLOSE_BRACE = re.compile(r"(?<!\s)\\\}")


@dataclass
class LintVerdict:
    """QA gate input: hard fail blocks the job, warnings ride along."""

    hard_fail: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _stash_math(markdown: str) -> tuple[str, list[str]]:
    """Replace every math span with a placeholder; return (text, spans)."""
    spans: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return _MATH_STASH.format(len(spans) - 1)

    return _MATH_SPAN.sub(_replace, markdown), spans


def _restore_math(markdown: str, spans: list[str]) -> str:
    """Put the untouched math spans back where their placeholders landed."""
    if not spans:
        return markdown
    return _MATH_RESTORE.sub(lambda match: spans[int(match.group(1))], markdown)


def normalize_markdown(markdown: str) -> str:
    """GFM normalization (syntax only — never source semantics).

    Math spans are protected verbatim across mdformat: the normalizer has no
    math extension and would otherwise escape LaTeX backslashes/underscores
    into invalid math.
    """
    protected, spans = _stash_math(markdown)
    normalized = mdformat.text(protected, extensions={"gfm"})
    return _restore_math(normalized, spans)


def _github_safe_span(span: str) -> str:
    """Rewrite one math span so GitHub's unescape keeps it valid math.

    - `\\text{old\\_way}`: draw the underscore instead of escaping it.
    - `\\{` / `\\}`: use the backslash-letter brace macros with padding.
    Idempotent: already-rewritten spans are returned unchanged.
    """
    rewritten = span
    previous = None
    while previous != rewritten:
        previous = rewritten
        rewritten = _UNDERSCORE_IN_TEXT.sub(
            lambda match: (
                r"\text{"
                + match.group(1)
                + "}"
                + _TEXT_UNDERSCORE_RULE
                + r"\text{"
                + match.group(2)
                + "}"
            ),
            rewritten,
        )
    rewritten = _OPEN_BRACE.sub(lambda _m: r"\lbrace ", rewritten)
    rewritten = _CLOSE_BRACE.sub(lambda _m: r" \rbrace", rewritten)
    return rewritten


def github_math_safe(markdown: str) -> str:
    """Output-time math transform for GitHub (see module docstring).

    Applied only to math spans; prose and every other byte stay untouched.
    """
    return _MATH_SPAN.sub(lambda match: _github_safe_span(match.group(0)), markdown)


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


__all__ = [
    "LintVerdict",
    "classify",
    "github_math_safe",
    "lint_markdown",
    "normalize_markdown",
]
