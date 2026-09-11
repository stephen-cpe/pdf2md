"""Deterministic coverage floor — token recall of OCR text in page markdown.

The LLM judge (A.3) scores coverage, but LLM-judging-LLM is self-congratulatory:
a fabricated page can score 99. This floor is the objective lower bound:
how many OCR tokens made it into the page markdown. Omission is measurable;
fabrication makes recall DROP, so it cannot hide behind a high judge score.

Token model (deterministic, no model calls):
- lowercase, strip Markdown syntax noise (fences/headers/links/emphasis/
  table pipes), then split on non-alphanumerics
- recall = |ocr tokens found in md tokens| / |ocr tokens|
- unmeasurable (None) when the OCR reference is too small to trust
  (empty, or under COVERAGE_FLOOR_MIN_OCR_TOKENS) — never gate on noise
"""

import re
from typing import Final

_TOKEN_SPLIT: Final = re.compile(r"[^a-z0-9]+")
_NOISE_PATTERNS: Final = tuple(
    re.compile(p)
    for p in (
        r"^```.*$",  # code fences
        r"^\s*#{1,6}\s",  # ATX heading markers (keep the text)
        r"!\[([^\]]*)\]\([^)]*\)",  # images -> keep the alt text, drop the path
        r"\[([^\]]*)\]\([^)]*\)",  # links -> keep the label text
        r"[*_`]+",  # emphasis/code markers
    )
)


def _strip_noise(line: str) -> str:
    """One line of markdown with syntax markers reduced to plain text."""
    for pattern in _NOISE_PATTERNS:
        if pattern.groups:
            line = pattern.sub(lambda m: m.group(1) or " ", line)
        else:
            line = pattern.sub(" ", line)
    return line


def tokenize(text: str) -> list[str]:
    """Deterministic alphanumeric tokens (order preserved, noise stripped)."""
    stripped_lines = (_strip_noise(line) for line in text.splitlines())
    lowered = "\n".join(stripped_lines).lower()
    return [t for t in _TOKEN_SPLIT.split(lowered) if t]


def recall(ocr_text: str, markdown: str, min_ocr_tokens: int = 30) -> float | None:
    """Fraction of OCR tokens present in markdown; None when unmeasurable.

    Args:
        ocr_text: the character ground truth for the page.
        markdown: the produced page markdown.
        min_ocr_tokens: OCR references smaller than this are noise —
            return None so callers never gate on a meaningless ratio.
    """
    ocr_tokens = tokenize(ocr_text)
    if len(ocr_tokens) < min_ocr_tokens:
        return None
    have = set(tokenize(markdown))
    hits = sum(1 for token in ocr_tokens if token in have)
    return hits / len(ocr_tokens)


def floor_failure(floor_score: float | None, coverage_floor: float) -> str | None:
    """Reason string when the floor is violated, else None.

    None floor_score (unmeasurable) never fails — the judge verdict alone
    stands for pages whose OCR reference was too small to measure.
    """
    if floor_score is None:
        return None
    if floor_score < coverage_floor:
        return (
            f"coverage floor not met: token recall {floor_score:.0%} < "
            f"{coverage_floor:.0%} (judge score alone is insufficient)"
        )
    return None


__all__ = ["floor_failure", "recall", "tokenize"]
