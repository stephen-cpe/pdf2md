"""Diagram -> Mermaid reinterpretation — the primary capability.

A cropped figure is sent to the Cloud vision agent, which decides whether it
can be faithfully re-expressed as Mermaid source. The candidate is then
validated deterministically (type allowlist + structural sanity) and, when
enabled, gated by a vision verifier that compares the Mermaid source against
the figure crop (reusing the A.3 verdict contract: coverage = fidelity).

Policy (tiered fallback, never silently drop a figure):
  1. mermaid — validated + verifier-approved reinterpretation
  2. table   — OCR/agent-grounded data table (data charts: bar/line/pie/...)
  3. image   — original asset + alt-text + caption (photos/illustrations/maps)

This module is pure logic plus one injected agent seam: `convert_figure`
calls the supplied converter/verifier callables. No I/O beyond the image path.
"""

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from src.pipeline.agent import AgentResult
from src.pipeline.envelope import EnvelopeError, ParsedDiagram, parse_diagram, parse_verdict
from src.pipeline.resilience import resilient

# Canonical Mermaid type identifiers as they appear on the first meaningful
# line of the source. `graph` is the legacy alias for `flowchart`.
_TYPE_ALIASES: dict[str, str] = {
    "graph": "flowchart",
    "flowchart": "flowchart",
    "sequencediagram": "sequenceDiagram",
    "classdiagram": "classDiagram",
    "statediagram": "stateDiagram-v2",
    "statediagram-v2": "stateDiagram-v2",
    "erdiagram": "erDiagram",
    "gantt": "gantt",
    "mindmap": "mindmap",
    "timeline": "timeline",
    "journey": "journey",
    "pie": "pie",
    "gitgraph": "gitGraph",
    "quadrantchart": "quadrantChart",
    "xychart-beta": "xychart-beta",
    "sankey-beta": "sankey-beta",
    "architecture-beta": "architecture-beta",
    "radar-beta": "radar-beta",
    "kanban": "kanban",
}

# Data charts: reinterpretation risks fabricated values, so the fallback is
# a grounded data table rather than a guessed pie/xychart.
_DATA_CHART_TYPES = frozenset({"pie", "xychart-beta", "sankey-beta", "radar-beta"})

_FENCE_OPEN = re.compile(r"^\s*```(?:mermaid)?\s*$", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"^\s*```\s*$")


class DiagramConverter(Protocol):
    """Minimal callable seam for the per-figure Cloud call (NFR-10)."""

    async def transcribe(
        self,
        image: Path | None,
        ocr_text: str,
        image_width: int,
        image_height: int,
        page_number: int = 1,
        rolling_context: str = "",
        outline: str = "",
    ) -> AgentResult:
        """Convert one figure; raises on transport failure."""
        ...  # pragma: no cover


@dataclass(frozen=True)
class DiagramData:
    """Tabular payload for a data chart (columns + string rows)."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def to_markdown_table(self) -> str:
        """Render a GFM pipe table; empty when no columns."""
        if not self.columns:
            return ""
        header = "| " + " | ".join(self.columns) + " |"
        sep = "| " + " | ".join("---" for _ in self.columns) + " |"
        body = [
            "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in self.rows
        ]
        return "\n".join([header, sep, *body])


@dataclass(frozen=True)
class DiagramResult:
    """One figure's reinterpretation outcome."""

    convertible: bool
    mermaid: str = ""
    diagram_type: str = ""
    confidence: int = 0
    description: str = ""
    data: DiagramData | None = None
    reason: str = ""


@dataclass(frozen=True)
class Representation:
    """Which representation to emit at the placeholder site + why."""

    kind: Literal["mermaid", "table", "image"]
    reason: str


def strip_fences(mermaid: str) -> str:
    """Drop a ```mermaid fence the model may have wrapped the source in."""
    lines = mermaid.splitlines()
    while lines and _FENCE_OPEN.match(lines[0]):
        lines.pop(0)
    while lines and _FENCE_CLOSE.match(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def _first_meaningful_line(mermaid: str) -> str:
    for line in mermaid.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("%%"):
            return stripped
    return ""


def detect_mermaid_type(mermaid: str) -> str | None:
    """Canonical Mermaid type from the first meaningful line, else None."""
    first = _first_meaningful_line(mermaid)
    if not first:
        return None
    token = first.split()[0].rstrip(":").lower()
    return _TYPE_ALIASES.get(token)


def validate_mermaid(mermaid: str, allowed: frozenset[str]) -> str | None:
    """Return a failure reason, or None when the source looks usable.

    Checks are deliberately structural, not a full Mermaid grammar: a valid
    type declaration on the first meaningful line, an allowlisted type, a
    non-empty body, and no leftover pipeline markers. A semantic fidelity
    check is the verifier's job (see `convert_figure`).
    """
    source = strip_fences(mermaid)
    if not source:
        return "empty Mermaid source"
    if "<<<" in source or "<!--FIG:" in source:
        return "Mermaid source contains leftover envelope/figure markers"
    detected = detect_mermaid_type(source)
    if detected is None:
        return f"unrecognized Mermaid diagram type: {_first_meaningful_line(source)[:40]!r}"
    if detected not in allowed:
        return f"Mermaid type {detected!r} not in allowlist"
    body = [
        line
        for line in source.splitlines()[1:]
        if line.strip() and not line.strip().startswith("%%")
    ]
    if not body:
        return f"{detected} diagram has no body"
    return None


def is_data_chart(diagram_type: str) -> bool:
    """True for chart types whose faithful fallback is a data table."""
    return diagram_type in _DATA_CHART_TYPES


def _parsed_to_result(parsed: ParsedDiagram) -> DiagramResult:
    data = None
    if parsed.data is not None:
        data = DiagramData(columns=parsed.data.columns, rows=parsed.data.rows)
    return DiagramResult(
        convertible=parsed.convertible,
        mermaid=strip_fences(parsed.mermaid),
        diagram_type=parsed.diagram_type,
        confidence=parsed.confidence,
        description=parsed.description,
        data=data,
    )


def _verifier_confidence(verdict_raw: AgentResult, coverage_floor: int) -> tuple[bool, int]:
    """Parse a fidelity verdict; malformed verdict = not approved."""
    try:
        verdict = parse_verdict(verdict_raw.raw)
    except EnvelopeError:
        return False, 0
    approved = verdict.verdict == "pass" and verdict.coverage >= coverage_floor
    return approved, verdict.coverage


async def convert_figure(
    converter: DiagramConverter,
    verifier: DiagramConverter | None,
    image: Path,
    grounding: str,
    *,
    page_number: int,
    width: int,
    height: int,
    allowed_types: frozenset[str],
    min_confidence: int,
    verify: bool = True,
) -> DiagramResult:
    """Reinterpret one figure crop as Mermaid; validate + optionally verify.

    The converter's self-reported confidence must clear `min_confidence`; when
    `verify` is set, a second vision call must also approve the candidate
    against the crop. Any failure yields a non-convertible result carrying a
    reason — the caller applies the tiered fallback, never dropping the figure.
    """
    try:
        raw = await resilient(
            _call_converter(converter, image, grounding, width, height, page_number),
            operation="diagram",
        )
        parsed = parse_diagram(raw.raw)
    except EnvelopeError as exc:
        return DiagramResult(convertible=False, reason=f"malformed diagram envelope: {exc.detail}")
    result = _parsed_to_result(parsed)
    if not result.convertible:
        return DiagramResult(
            convertible=False,
            diagram_type=result.diagram_type,
            confidence=result.confidence,
            description=result.description,
            data=result.data,
            reason="agent reported not convertible",
        )
    if result.confidence < min_confidence:
        return DiagramResult(
            convertible=False,
            diagram_type=result.diagram_type,
            confidence=result.confidence,
            description=result.description,
            data=result.data,
            reason=f"confidence {result.confidence} < {min_confidence}",
        )
    problem = validate_mermaid(result.mermaid, allowed_types)
    if problem is not None:
        return DiagramResult(
            convertible=False,
            diagram_type=result.diagram_type,
            confidence=result.confidence,
            description=result.description,
            data=result.data,
            reason=problem,
        )
    if verify and verifier is not None:
        approved, coverage = await _run_verify(
            verifier, image, result.mermaid, grounding, width, height, page_number
        )
        if not approved:
            return DiagramResult(
                convertible=False,
                diagram_type=result.diagram_type,
                confidence=result.confidence,
                description=result.description,
                data=result.data,
                reason=f"verifier rejected Mermaid (fidelity {coverage})",
            )
        result = DiagramResult(
            convertible=True,
            mermaid=result.mermaid,
            diagram_type=detect_mermaid_type(result.mermaid) or result.diagram_type,
            confidence=max(result.confidence, coverage),
            description=result.description,
            data=result.data,
        )
    else:
        result = DiagramResult(
            convertible=True,
            mermaid=result.mermaid,
            diagram_type=detect_mermaid_type(result.mermaid) or result.diagram_type,
            confidence=result.confidence,
            description=result.description,
            data=result.data,
        )
    return result


def _call_converter(
    converter: DiagramConverter,
    image: Path,
    grounding: str,
    width: int,
    height: int,
    page_number: int,
) -> Callable[[], Awaitable[AgentResult]]:
    async def _run() -> AgentResult:
        return await converter.transcribe(image, grounding, width, height, page_number, "", "")

    return _run


async def _run_verify(
    verifier: DiagramConverter,
    image: Path,
    mermaid: str,
    grounding: str,
    width: int,
    height: int,
    page_number: int,
) -> tuple[bool, int]:
    candidate = f"FIGURE GROUNDING:\n{grounding}\nCANDIDATE MERMAID:\n{mermaid}"

    async def _run() -> AgentResult:
        return await verifier.transcribe(image, candidate, width, height, page_number, "", "")

    raw = await resilient(_run, operation="diagram-verify")
    return _verifier_confidence(raw, 0)


def choose_representation(
    result: DiagramResult,
    *,
    fallback: Literal["image", "table", "both"] = "both",
    image_available: bool = True,
) -> Representation:
    """Tiered decision: mermaid → table (charts) → image."""
    if result.convertible and result.mermaid:
        return Representation(kind="mermaid", reason=f"converted ({result.diagram_type})")
    chart = is_data_chart(result.diagram_type)
    wants_table = fallback in ("table", "both") and chart and result.data is not None
    if wants_table:
        return Representation(
            kind="table", reason=f"data chart fallback: {result.reason or 'chart'}"
        )
    if image_available:
        return Representation(kind="image", reason=result.reason or "not convertible")
    return Representation(kind="image", reason=result.reason or "image unavailable")


__all__ = [
    "DiagramConverter",
    "DiagramData",
    "DiagramResult",
    "Representation",
    "choose_representation",
    "convert_figure",
    "detect_mermaid_type",
    "is_data_chart",
    "strip_fences",
    "validate_mermaid",
]
