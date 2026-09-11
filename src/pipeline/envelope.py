"""Textual-envelope parser — deterministic extraction + validation.

SRS Appendix A.2/A.3: Ollama Cloud has no structured outputs, so this parser
IS the contract enforcement. Rules, all strict on purpose (a silent
mis-parse corrupts the pipeline worse than a loud retry):
- Section markers match at column 0, exact, case-sensitive. Indented or
  mid-line lookalikes are CONTENT, never markers (A.2 "line-start" rule).
- Every required section must open once and close once; duplicates, stray
  ends, truncation, and missing sections are EnvelopeError.
- FIGURES/FURNITURE/VERDICT payloads are JSON validated AFTER extraction,
  so one unescaped quote in Markdown can never invalidate the response.

Malformed input never discards data: EnvelopeError carries the verbatim raw
response (caller preserves it + flags needs_review after the retry cap) and
corrective_prompt() builds the retry instruction. Cap: 2 attempts total.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Literal

MAX_ENVELOPE_ATTEMPTS = 2

TRANSCRIPTION_SECTIONS = ("MARKDOWN", "FIGURES", "FURNITURE", "NOTES")
VERDICT_SECTIONS = ("VERDICT",)
QA_SECTIONS = ("PATCHES", "SUMMARY")


def _open(tag: str) -> str:
    return f"<<<{tag}>>>"


def _close(tag: str) -> str:
    return f"<<<END_{tag}>>>"


class EnvelopeError(ValueError):
    """Malformed envelope; .raw preserves the verbatim model response."""

    def __init__(self, detail: str, raw: str) -> None:
        super().__init__(detail)
        self.detail = detail
        self.raw = raw


@dataclass(frozen=True)
class Figure:
    """One FIGURES entry: absolute-pixel bbox + alt/caption (FR-AGT-3)."""

    index: int
    bbox: tuple[float, float, float, float]
    alt: str
    caption: str | None


@dataclass(frozen=True)
class Furniture:
    """Running furniture, excluded from body (FR-AGT-4)."""

    header: str = ""
    footer: str = ""
    page_number: str = ""


@dataclass(frozen=True)
class ParsedEnvelope:
    """Validated transcription envelope (A.2)."""

    markdown: str
    figures: tuple[Figure, ...] = ()
    furniture: Furniture = field(default_factory=Furniture)
    notes: str = ""


@dataclass(frozen=True)
class ParsedVerdict:
    """Validated verification verdict (A.3)."""

    coverage: int
    misses: tuple[str, ...] = ()
    structure_issues: tuple[str, ...] = ()
    verdict: Literal["pass", "retry"] = "retry"


@dataclass(frozen=True)
class Patch:
    """One section-anchored find/replace operation (A.4)."""

    section: str
    find: str
    replace: str


@dataclass(frozen=True)
class ParsedQa:
    """Validated whole-document QA response (A.4)."""

    patches: tuple[Patch, ...] = ()
    summary: str = ""


def _extract_sections(raw: str, required: tuple[str, ...]) -> dict[str, str]:
    """Line-anchored section extraction; raises EnvelopeError on any defect."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for lineno, line in enumerate(raw.splitlines(), start=1):
        # Column-0, exact, case-sensitive (trailing whitespace tolerated).
        # Anything else — indented, mid-line, or unknown tag — is CONTENT.
        if line.startswith("<<<"):
            tag = line.rstrip()[3:]
            if tag.endswith(">>>"):
                name = tag[:-3]
                if name in required:
                    if name in sections:
                        raise EnvelopeError(f"line {lineno}: duplicate section {line!r}", raw)
                    current = name
                    sections[name] = []
                    continue
                if name.startswith("END_") and name[4:] in required:
                    target = name[4:]
                    if current != target:
                        raise EnvelopeError(f"line {lineno}: unexpected {line!r}", raw)
                    current = None
                    continue
                if current is None:
                    raise EnvelopeError(f"line {lineno}: stray marker {line!r}", raw)
        if current is None:
            if line.strip():
                raise EnvelopeError(
                    f"line {lineno}: content outside any section: {line[:60]!r}", raw
                )
            continue
        sections[current].append(line)
    if current is not None:
        raise EnvelopeError(f"truncated: {_open(current)} never closed", raw)
    missing = [tag for tag in required if tag not in sections]
    if missing:
        raise EnvelopeError(f"missing sections: {missing}", raw)
    return {tag: "\n".join(sections[tag]).strip() for tag in required}


def _payload_json(section: str, text: str, raw: str) -> Any:
    try:
        return json.loads(text) if text else (_default_payload(section))
    except json.JSONDecodeError as exc:
        raise EnvelopeError(f"invalid {section} JSON: {exc}", raw) from exc


def _default_payload(section: str) -> Any:
    return [] if section in ("FIGURES",) else {}


def _parse_figures(text: str, raw: str) -> tuple[Figure, ...]:
    payload = _payload_json("FIGURES", text, raw)
    if not isinstance(payload, list):
        raise EnvelopeError("FIGURES payload must be a JSON array", raw)
    figures: list[Figure] = []
    for pos, item in enumerate(payload):
        if not isinstance(item, dict):
            raise EnvelopeError(f"FIGURES[{pos}] must be an object", raw)
        try:
            index = int(item["index"])
            bbox_raw = item["bbox"]
            alt = item["alt"]
            caption = item.get("caption")
        except KeyError as exc:
            raise EnvelopeError(f"FIGURES[{pos}] missing key {exc}", raw) from exc
        if (
            not isinstance(bbox_raw, list)
            or len(bbox_raw) != 4
            or not all(isinstance(v, (int, float)) for v in bbox_raw)
        ):
            raise EnvelopeError(f"FIGURES[{pos}].bbox must be 4 numbers", raw)
        if not isinstance(alt, str):
            raise EnvelopeError(f"FIGURES[{pos}].alt must be a string", raw)
        if caption is not None and not isinstance(caption, str):
            raise EnvelopeError(f"FIGURES[{pos}].caption must be a string or null", raw)
        x0, y0, x1, y1 = (float(v) for v in bbox_raw)
        figures.append(Figure(index=index, bbox=(x0, y0, x1, y1), alt=alt, caption=caption))
    return tuple(figures)


def _parse_furniture(text: str, raw: str) -> Furniture:
    payload = _payload_json("FURNITURE", text, raw)
    if not isinstance(payload, dict):
        raise EnvelopeError("FURNITURE payload must be a JSON object", raw)
    return Furniture(
        header=str(payload.get("header", "") or ""),
        footer=str(payload.get("footer", "") or ""),
        page_number=str(payload.get("page-number", "") or ""),
    )


def parse_transcription(raw: str) -> ParsedEnvelope:
    """Parse + validate a transcription response (A.2)."""
    sections = _extract_sections(raw, TRANSCRIPTION_SECTIONS)
    return ParsedEnvelope(
        markdown=sections["MARKDOWN"],
        figures=_parse_figures(sections["FIGURES"], raw),
        furniture=_parse_furniture(sections["FURNITURE"], raw),
        notes=sections["NOTES"],
    )


def parse_verdict(raw: str) -> ParsedVerdict:
    """Parse + validate a verification response (A.3, FR-AGT-3a)."""
    sections = _extract_sections(raw, VERDICT_SECTIONS)
    payload = _payload_json("VERDICT", sections["VERDICT"], raw)
    if not isinstance(payload, dict):
        raise EnvelopeError("VERDICT payload must be a JSON object", raw)
    coverage = payload.get("coverage")
    if isinstance(coverage, bool) or not isinstance(coverage, int) or not 0 <= coverage <= 100:
        raise EnvelopeError("VERDICT.coverage must be an integer 0-100", raw)
    misses = payload.get("misses", [])
    issues = payload.get("structure_issues", [])
    if not isinstance(misses, list) or not all(isinstance(m, str) for m in misses):
        raise EnvelopeError("VERDICT.misses must be a string array", raw)
    if not isinstance(issues, list) or not all(isinstance(m, str) for m in issues):
        raise EnvelopeError("VERDICT.structure_issues must be a string array", raw)
    verdict = payload.get("verdict")
    if verdict not in ("pass", "retry"):
        raise EnvelopeError('VERDICT.verdict must be "pass" or "retry"', raw)
    return ParsedVerdict(
        coverage=coverage,
        misses=tuple(misses),
        structure_issues=tuple(issues),
        verdict=verdict,
    )


def parse_qa(raw: str) -> ParsedQa:
    """Parse + validate a whole-document QA response (A.4)."""
    sections = _extract_sections(raw, QA_SECTIONS)
    payload = _payload_json("PATCHES", sections["PATCHES"], raw)
    if not isinstance(payload, list):
        raise EnvelopeError("PATCHES payload must be a JSON array", raw)
    patches: list[Patch] = []
    for pos, item in enumerate(payload):
        if not isinstance(item, dict):
            raise EnvelopeError(f"PATCHES[{pos}] must be an object", raw)
        try:
            section, find, replace = item["section"], item["find"], item["replace"]
        except KeyError as exc:
            raise EnvelopeError(f"PATCHES[{pos}] missing key {exc}", raw) from exc
        if not all(isinstance(v, str) for v in (section, find, replace)) or not find:
            raise EnvelopeError(
                f"PATCHES[{pos}] needs string section/find/replace with non-empty find", raw
            )
        patches.append(Patch(section=section, find=find, replace=replace))
    return ParsedQa(patches=tuple(patches), summary=sections["SUMMARY"])


def corrective_prompt(
    error: EnvelopeError, required: tuple[str, ...] = TRANSCRIPTION_SECTIONS
) -> str:
    """Retry instruction naming the defect (injected on corrective retry)."""
    expected = " ".join(_open(tag) + " … " + _close(tag) for tag in required)
    return (
        "Your previous response was malformed and rejected: "
        f"{error.detail}. Re-emit the COMPLETE envelope — markers at column 0, "
        f"exact and case-sensitive, every section opened once and closed once: {expected} "
        "Payload sections must contain valid JSON. Do not explain; output only the envelope."
    )


__all__ = [
    "MAX_ENVELOPE_ATTEMPTS",
    "TRANSCRIPTION_SECTIONS",
    "VERDICT_SECTIONS",
    "EnvelopeError",
    "Figure",
    "Furniture",
    "ParsedEnvelope",
    "ParsedQa",
    "ParsedVerdict",
    "Patch",
    "corrective_prompt",
    "parse_qa",
    "parse_transcription",
    "parse_verdict",
]
