"""Hybrid routing — deterministic extraction first, agentic on doubt.

FR-PDF-5 (Phase 3): born-digital pages skip the OCR+transcribe stage
when three signals agree the page is safe:
  1. native text exists (PyMuPDF text layer, threshold-gated)
  2. deterministic structure extraction (pymupdf4llm) succeeds and
     stays within a token budget
  3. the existing verifier agent approves the deterministic candidate
     (one Cloud call — not zero: the verifier is the quality gate that
     catches flat tables and lost structure)

Any objection → the page takes the full agentic path unchanged. Output
is byte-identical to the agentic path from the pipeline's viewpoint:
a verified page is a verified page.

Cost model: agentic page = render + OCR + transcribe + verify.
Deterministic page = render + verify. The verifier is kept for BOTH
paths because it is the only stage that can see the page image and
compare it against the candidate.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# Deterministic extraction budget: pymupdf4llm can explode on pages with
# pathological layouts (degenerate tables). Anything beyond this is a
# routing failure → escalate, never crash or truncate silently.
MAX_DETERMINISTIC_CHARS = 60_000
# Native text floor (words): below this the page is scanned/sparse —
# not a deterministic-extraction candidate regardless of anything else.
MIN_NATIVE_WORDS = 20
# Deterministic extraction drops raster image data (picture text only).
# Small placed images (header logos, bullets, watermarks) are furniture;
# substantial placed images are content figures the agentic path must
# resolve. A page routes deterministic only when total placed image area
# stays under this fraction of the page area.
MAX_DECORATIVE_IMAGE_AREA = 0.03


def _placed_image_area_ratio(pdf_path: Path, page_number: int) -> float:
    """Total placed raster area / page area (1.0 when unknown/undecidable)."""
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_number - 1]
        page_area = abs(page.rect)
        if page_area <= 0:
            return 1.0
        total = 0.0
        for entry in page.get_images(full=True):
            try:
                rects = page.get_image_rects(entry[0])
            except Exception:  # noqa: BLE001 - unplaced/odd xref: conservative
                return 1.0
            total += sum(abs(rect) for rect in rects)
        return float(total / page_area)
    finally:
        doc.close()


@dataclass(frozen=True)
class RouteDecision:
    """One page's routing verdict + why (report/diagnostics)."""

    deterministic: bool
    reason: str
    native_words: int
    native_text: str = ""


def native_text(pdf_path: Path, page_number: int) -> str:
    """Raw native text layer of one 1-based page ('' when none)."""
    doc = pymupdf.open(pdf_path)
    try:
        if page_number < 1 or page_number > doc.page_count:
            raise ValueError(f"page {page_number} out of range (1..{doc.page_count})")
        return str(doc[page_number - 1].get_text())
    finally:
        doc.close()


def native_word_count(pdf_path: Path, page_number: int) -> int:
    """Word count of the native text layer (0 when absent)."""
    return len(native_text(pdf_path, page_number).split())


def extract_deterministic(pdf_path: Path, page_number: int) -> str | None:
    """Structure-aware markdown for one 1-based page; None on any defect.

    pymupdf4llm's `pages` parameter is 0-BASED (verified against 1.28.x
    layout path) — this module is 1-based throughout like the rest of
    the project. headers/footers handling can choke on odd PDFs; every
    failure mode here routes to the agentic path (the safe default),
    never raises, never truncates.
    """
    try:
        import pymupdf4llm  # type: ignore[import-untyped]

        markdown = pymupdf4llm.to_markdown(str(pdf_path), pages=[page_number - 1])
    except Exception:  # noqa: BLE001 - routing must never crash the page
        return None
    if not markdown or not markdown.strip():
        return None
    if len(markdown) > MAX_DETERMINISTIC_CHARS:
        return None
    # Degenerate marker: extraction emitted the whole file instead of the
    # requested page (defensive against pymupdf4llm version drift).
    return markdown if _looks_single_page(markdown, pdf_path, page_number) else None


def _looks_single_page(markdown: str, pdf_path: Path, page_number: int) -> bool:
    """Reject extractions that clearly contain other pages' content."""
    doc = pymupdf.open(pdf_path)
    try:
        total = doc.page_count
        # Extract neighboring page text fingerprints (short prefixes).
        neighbor_prefixes = []
        for neighbor in (page_number - 1, page_number + 1):
            if 1 <= neighbor <= total and neighbor != page_number:
                prefix = _word_fingerprint(doc[neighbor - 1].get_text())
                if len(prefix) >= 5:
                    neighbor_prefixes.append(prefix)
        own_prefix = _word_fingerprint(doc[page_number - 1].get_text())
    finally:
        doc.close()
    tokens = _word_fingerprint(markdown)
    # A neighbor's fingerprint appearing wholesale in this page's extraction
    # means the extraction ignored pages=[...] — escalate.
    for prefix in neighbor_prefixes:
        if _contains_run(tokens, prefix):
            return False
    # Sanity: own page text should overlap the extraction at least
    # partially (extraction reorders/reformats, so containment is fuzzy).
    own_in_md = sum(1 for word in set(own_prefix) if word in set(tokens))
    return own_in_md >= max(1, len(set(own_prefix)) // 4)


def _word_fingerprint(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _contains_run(haystack: list[str], needle: list[str]) -> bool:
    """True when needle's words appear as a contiguous run in haystack."""
    if not needle or len(needle) > len(haystack):
        return False
    first = needle[0]
    window = len(needle)
    for start in range(len(haystack) - window + 1):
        if haystack[start] == first and haystack[start : start + window] == needle:
            return True
    return False


def decide_route(
    pdf_path: Path,
    page_number: int,
    *,
    min_native_words: int = MIN_NATIVE_WORDS,
) -> RouteDecision:
    """Cheap pre-check (no LLM): is this page a deterministic candidate?

    The verifier veto happens later (pages.py) because only there do we
    have the agent seam. This function answers: "is deterministic
    extraction even worth trying?" Corrupt/undecodable PDFs answer no —
    routing must never crash the page.
    """
    try:
        text = native_text(pdf_path, page_number)
    except Exception:  # noqa: BLE001 - unreadable page: agentic path decides
        text = ""
    words = len(text.split())
    if words < min_native_words:
        return RouteDecision(
            deterministic=False,
            reason=f"native text too sparse ({words} words < {min_native_words})",
            native_words=words,
            native_text=text,
        )
    try:
        image_area = _placed_image_area_ratio(pdf_path, page_number)
    except Exception:  # noqa: BLE001 - undecidable geometry: agentic path
        image_area = 1.0
    if image_area > MAX_DECORATIVE_IMAGE_AREA:
        return RouteDecision(
            deterministic=False,
            reason=(
                f"placed image area {image_area:.0%} exceeds decorative cap "
                f"({MAX_DECORATIVE_IMAGE_AREA:.0%}) — figures need agentic resolution"
            ),
            native_words=words,
            native_text=text,
        )
    return RouteDecision(
        deterministic=True,
        reason=f"native text present ({words} words), images decorative ({image_area:.0%} area)",
        native_words=words,
        native_text=text,
    )


def route_page(
    pdf_path: Path,
    page_number: int,
    *,
    min_native_words: int = MIN_NATIVE_WORDS,
) -> tuple[RouteDecision, str | None]:
    """Decide + extract in one call; (decision, candidate-markdown).

    candidate is None whenever the decision is non-deterministic OR the
    deterministic extraction itself failed — both mean: go agentic.
    """
    decision = decide_route(pdf_path, page_number, min_native_words=min_native_words)
    if not decision.deterministic:
        return decision, None
    candidate = extract_deterministic(pdf_path, page_number)
    if candidate is None:
        return (
            RouteDecision(
                deterministic=False,
                reason="deterministic extraction failed or over budget",
                native_words=decision.native_words,
            ),
            None,
        )
    return decision, candidate


__all__ = [
    "MAX_DECORATIVE_IMAGE_AREA",
    "MAX_DETERMINISTIC_CHARS",
    "MIN_NATIVE_WORDS",
    "RouteDecision",
    "decide_route",
    "extract_deterministic",
    "native_text",
    "native_word_count",
    "route_page",
]
