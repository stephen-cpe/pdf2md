"""Whole-document QA — patch application + Cloud QA runner.

Patches are section-anchored find/replace ops (A.4). Application is
deterministic: the anchor is the first heading whose normalized text equals
the section; `find` must occur EXACTLY ONCE in the region from that anchor
to the next heading of equal-or-higher level. Anything else is rejected and
logged — applied patches are reversible from the log (swap find/replace).
A malformed QA envelope means no patches (logged), never a job failure.
"""

import functools
import re
from dataclasses import dataclass, field

from src.pipeline.agent import AgentResult, TranscriptionAgent
from src.pipeline.envelope import EnvelopeError, ParsedQa, Patch, parse_qa
from src.pipeline.resilience import resilient

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class PatchLog:
    """One patch decision — the reversibility record (6.3 DoD)."""

    section: str
    find: str
    replace: str
    applied: bool
    reason: str = ""


@dataclass
class QaOutcome:
    """QA result: patched markdown + full decision log + agent summary."""

    markdown: str
    applied: list[PatchLog] = field(default_factory=list)
    rejected: list[PatchLog] = field(default_factory=list)
    summary: str = ""


def _normalize_heading(line: str) -> str:
    if m := _HEADING.match(line.strip()):
        return m.group(2).strip().casefold()
    return line.strip().casefold()


def _anchor_region(lines: list[str], section: str) -> tuple[int, int] | None:
    """(start, end) line range for a section anchor, or None when absent."""
    wanted = section.strip().casefold()
    anchor = level = None
    for pos, line in enumerate(lines):
        if (m := _HEADING.match(line)) and m.group(2).strip().casefold() == wanted:
            anchor, level = pos, len(m.group(1))
            break
    if anchor is None:
        return None
    end = len(lines)
    for pos in range(anchor + 1, len(lines)):
        if (m := _HEADING.match(lines[pos])) and len(m.group(1)) <= (level or 6):
            end = pos
            break
    return anchor, end


def apply_patches(markdown: str, patches: list[Patch]) -> QaOutcome:
    """Apply patches deterministically; reject ambiguous/missing anchors."""
    lines = markdown.splitlines()
    applied: list[PatchLog] = []
    rejected: list[PatchLog] = []
    for patch in patches:
        region = _anchor_region(lines, patch.section)
        if region is None:
            rejected.append(
                PatchLog(patch.section, patch.find, patch.replace, False, "no such section")
            )
            continue
        start, end = region
        # Occurrence count across the region, excluding the anchor heading
        # line itself: exactly-1 keeps application deterministic.
        total = 0
        target = -1
        for pos in range(start, end):
            if pos == start and patch.find in lines[start]:
                continue
            occurrences = lines[pos].count(patch.find)
            if occurrences:
                total += occurrences
                target = pos
        if total != 1:
            rejected.append(
                PatchLog(
                    patch.section,
                    patch.find,
                    patch.replace,
                    False,
                    f"find occurs {total}x in section (need exactly 1)",
                )
            )
            continue
        lines[target] = lines[target].replace(patch.find, patch.replace, 1)
        applied.append(PatchLog(patch.section, patch.find, patch.replace, True))
    trailing = "\n" if markdown.endswith("\n") else ""
    return QaOutcome(markdown="\n".join(lines) + trailing, applied=applied, rejected=rejected)


async def run_qa_pass(agent: TranscriptionAgent, markdown: str, outline: str) -> QaOutcome:
    """Whole-document agent QA (A.4, FR-QA-2): fetch patches, apply, log."""
    user_content = (
        "Assembled document outline:\n" + (outline or "none") + "\n\n"
        "ASSEMBLED MARKDOWN:\n" + markdown
    )
    raw: AgentResult = await resilient(
        functools.partial(agent.transcribe, None, user_content, 0, 0),
        operation="final-qa",
    )
    try:
        parsed: ParsedQa = parse_qa(raw.raw)
    except EnvelopeError as exc:
        return QaOutcome(markdown=markdown, summary=f"QA envelope malformed, no patches: {exc}")
    outcome = apply_patches(markdown, list(parsed.patches))
    outcome.summary = parsed.summary
    return outcome


__all__ = ["PatchLog", "QaOutcome", "apply_patches", "run_qa_pass"]
