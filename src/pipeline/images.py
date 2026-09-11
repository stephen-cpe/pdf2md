"""Figure handling — resolve placeholders, caption, integrity.

5.1 resolve: native PDF objects whose placement matches the agent box
    (original bytes, downscaled to a 1600px long edge when larger),
    crop-from-render fallback using the agent bbox.
    Every resolved figure is saved under
    assets/ (FR-IMG-2 names) + recorded in the images table.
5.2 caption: ![alt<=125](assets/…) + verbatim italic caption (FR-IMG-3).
5.3 details: config-gated <details> long description (FR-IMG-4, default on).
5.4 integrity: link↔file bijection + no leftover placeholders (FR-IMG-6,
    FR-QA-3 hard-fail input). Unresolvable placeholders are LEFT in place
    so QA fails loudly — never silently dropped (FR-AGT-3).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import pymupdf
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import repo
from src.db.models import ImageSource
from src.pdf import (
    ExtractedImage,
    asset_name,
    crop_from_render,
    extract_native_images,
    render_filename,
)

# Two accepted token shapes: the SRS form <!--FIG:page:INDEX:x0,y0,x1,y1-->
# (what the prompt specifies; INDEX is 1,2,3... — never the literal "idx")
# and the numeric <!--FIG:<page>:<idx>:...--> shape without the literal.
# In the SRS form the embedded number is the figure
# INDEX (placement follows where the token sits).
FIG_PATTERN = re.compile(r"<!--FIG:(?:page:(\d+)|(\d+):(\d+)):([\d.,]+)-->")
# Any FIG-looking comment, well-formed or not (e.g. a literal
# <!--FIG:page:idx:...--> where the agent copied the prompt's INDEX
# placeholder verbatim). parse_placeholders() ignores malformed tokens for
# resolution, but check_assets() must still fail them loudly — never silent.
FIG_ANY_PATTERN = re.compile(r"<!--FIG:.*?-->")
FIG_MARKER = "<!--FIG:"
LINK_PATTERN = re.compile(r"\[[^\]]*\]\((assets/[^)\s]+)\)")
MAX_ALT_CHARS = 125
MAX_NATIVE_DIM = 1600  # long edge cap for native-sourced saves (GitHub sanity)
_NATIVE_JPEG_QUALITY = 85

# Geometry-match gates for native pairing (5.1): a native object backs a
# placeholder only when it sits where the agent pointed AND is roughly the
# same size. Containment alone is not enough — a 12pt badge icon placed
# inside a flowchart region (arxiv p4) fully contains in the small box yet
# is 0.3% of the agent box's area; pairing it would ship the icon with the
# flowchart's caption and drop the real figure.
_GEO_CONTAIN_MIN = 0.5  # intersection / smaller-box area
_GEO_AREA_MIN = 0.25  # native_displayed / agent_box (rejects tiny icons)
_GEO_AREA_MAX = 4.0  # (rejects page-spanning unions vs one figure)
_GEO_FALLBACK_DPI = 200  # when the render PNG is missing (tests/edge cases)


@dataclass(frozen=True)
class FigRef:
    """One placeholder token (FR-AGT-3): agent index + absolute-pixel bbox.

    `page` is the embedded number for older numeric tokens, None for the SRS form
    (placement follows token position, never the embedded number).
    """

    page: int | None
    index: int
    bbox: tuple[float, float, float, float]
    token: str


@dataclass(frozen=True)
class FigurePayload:
    """Agent FIGURES entry (from pages.omissions figures)."""

    index: int
    bbox: tuple[float, float, float, float]
    alt: str
    caption: str | None = None


@dataclass
class AssetReport:
    """Integrity verdict (5.4): ok only on full bijection, no leftovers."""

    ok: bool
    missing_files: list[str] = field(default_factory=list)
    orphan_files: list[str] = field(default_factory=list)
    leftover_placeholders: list[str] = field(default_factory=list)


@dataclass
class ResolveResult:
    """resolve_page_figures outcome: updated markdown + placement log."""

    markdown: str
    resolved: int = 0
    orphaned: list[dict[str, object]] = field(default_factory=list)


def parse_placeholders(markdown: str) -> list[FigRef]:
    """All FIG tokens in document order (malformed tokens ignored here)."""
    refs: list[FigRef] = []
    for match in FIG_PATTERN.finditer(markdown):
        try:
            coords = tuple(float(v) for v in match.group(4).split(","))
            if len(coords) != 4:
                continue
            x0, y0, x1, y1 = coords
        except ValueError:
            continue
        if match.group(1) is not None:
            page, index = None, int(match.group(1))
        else:
            page, index = int(match.group(2)), int(match.group(3))
        refs.append(FigRef(page=page, index=index, bbox=(x0, y0, x1, y1), token=match.group(0)))
    return refs


def _rect_area(box: tuple[float, float, float, float]) -> float:
    """Area, floored at 0 for degenerate boxes."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _rect_inter(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    """Intersection area of two x0,y0,x1,y1 boxes."""
    return _rect_area(
        (
            max(left[0], right[0]),
            max(left[1], right[1]),
            min(left[2], right[2]),
            min(left[3], right[3]),
        )
    )


def _native_match_score(
    ref_px: tuple[float, float, float, float], nat_px: tuple[float, float, float, float]
) -> float | None:
    """IoU match score for one agent box vs one native placement (render px).

    Returns None when incompatible: the smaller box must lie mostly inside
    the larger AND the displayed areas must be within 4x of each other.
    """
    ref_area = _rect_area(ref_px)
    nat_area = _rect_area(nat_px)
    if ref_area <= 0.0 or nat_area <= 0.0:
        return None
    inter = _rect_inter(ref_px, nat_px)
    if inter / min(ref_area, nat_area) < _GEO_CONTAIN_MIN:
        return None
    ratio = nat_area / ref_area
    if not (_GEO_AREA_MIN <= ratio <= _GEO_AREA_MAX):
        return None
    return inter / (ref_area + nat_area - inter)


def _render_size(
    render_path: Path, page_width_pts: float, page_height_pts: float
) -> tuple[int, int]:
    """Render PNG dimensions; falls back to an assumed DPI when missing."""
    try:
        probe = pymupdf.Pixmap(str(render_path))  # type: ignore[no-untyped-call]
        return probe.width, probe.height
    except Exception:  # noqa: BLE001 - probe fallback: any unreadable render uses assumed DPI
        scale = _GEO_FALLBACK_DPI / 72.0
        return round(page_width_pts * scale), round(page_height_pts * scale)


def match_natives_by_geometry(
    ref_bboxes: list[tuple[float, float, float, float]],
    natives: list[ExtractedImage],
    pdf_path: Path,
    page_number: int,
    render_path: Path,
) -> dict[int, int]:
    """Pair agent boxes to native objects by placement geometry.

    Returns {ref_position: native_position}; unmatched refs must crop from
    the render. Each native is used at most once (best IoU wins). A native
    without placement info never pairs — cropping its region is always
    content-correct, a wrong native never is.
    """
    placed = [(pos, img) for pos, img in enumerate(natives) if img.bbox is not None]
    if not ref_bboxes or not placed:
        return {}
    try:
        doc = pymupdf.open(pdf_path)  # type: ignore[no-untyped-call]
    except OSError, ValueError, RuntimeError:
        return {}
    try:
        if page_number > doc.page_count:
            return {}
        rect = doc[page_number - 1].rect
        page_w, page_h = rect.width, rect.height
    finally:
        doc.close()  # type: ignore[no-untyped-call]
    if page_w <= 0 or page_h <= 0:
        return {}
    render_w, render_h = _render_size(render_path, page_w, page_h)
    if render_w <= 0 or render_h <= 0:
        return {}
    scale_x, scale_y = render_w / page_w, render_h / page_h

    def _to_px(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return (box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y)

    taken: set[int] = set()
    matches: dict[int, int] = {}
    for ref_pos, ref_box in enumerate(ref_bboxes):
        best: tuple[float, int] | None = None
        for nat_pos, img in placed:
            if nat_pos in taken or img.bbox is None:
                continue
            score = _native_match_score(ref_box, _to_px(img.bbox))
            if score is not None and (best is None or score > best[0]):
                best = (score, nat_pos)
        if best is not None:
            taken.add(best[1])
            matches[ref_pos] = best[1]
    return matches


def shorten_alt(alt: str) -> str:
    """Concise alt-text ≤125 chars (FR-IMG-3); full text lives in <details>."""
    text = " ".join(alt.split())
    if len(text) <= MAX_ALT_CHARS:
        return text
    return text[: MAX_ALT_CHARS - 1].rstrip() + "…"


def cap_native_pixels(data: bytes, ext: str) -> tuple[str, bytes]:
    """Downscale an oversized native raster to MAX_NATIVE_DIM (long edge).

    A 12pt badge icon saved at 3886px renders absurdly large on GitHub;
    the placed size never justifies more than this. At-or-under images
    return byte-identical (JPEG passthrough stays verbatim). Format is
    preserved (jpg→JPEG, else PNG). Unreadable bytes pass through untouched
    — capping must never fail a figure.
    """
    from io import BytesIO

    from PIL import Image

    try:
        with Image.open(BytesIO(data)) as img:
            width, height = img.size
            if max(width, height) <= MAX_NATIVE_DIM:
                return ext, data
            scale = MAX_NATIVE_DIM / max(width, height)
            resized = img.resize(
                (round(width * scale), round(height * scale)), Image.Resampling.LANCZOS
            )
            out = BytesIO()
            if ext == "jpg":
                resized = resized.convert("RGB")
                resized.save(out, format="JPEG", quality=_NATIVE_JPEG_QUALITY)
                return "jpg", out.getvalue()
            resized.save(out, format="PNG")
            return "png", out.getvalue()
    except Exception:  # noqa: BLE001 - capping must never fail a figure
        return ext, data


def figure_block(filename: str, alt: str, caption: str | None, fig_details: bool) -> str:
    """Image link + verbatim caption + optional long description (5.2/5.3)."""
    short = shorten_alt(alt)
    parts = [f"![{short}](assets/{filename})"]
    if caption:
        parts.append(f"*{caption}*")
    if fig_details and " ".join(alt.split()) != short:
        parts.append(
            "<details>\n<summary>Image description</summary>\n\n"
            + " ".join(alt.split())
            + "\n</details>"
        )
    return "\n\n".join(parts)


async def _save_figure(
    session: AsyncSession,
    *,
    job_id: UUID,
    page_number: int,
    index: int,
    bbox: tuple[float, float, float, float],
    alt: str,
    caption: str | None,
    blob: tuple[str, bytes] | None,
    render_path: Path,
    assets_dir: Path,
) -> tuple[str, ImageSource] | None:
    """Extract (native bytes or render crop) + record row. None when impossible."""
    filename: str | None = None
    source = ImageSource.CROP
    try:
        if blob is not None:
            ext, data = cap_native_pixels(blob[1], blob[0])
            filename = asset_name(page_number, index, ext)
            (assets_dir / filename).write_bytes(data)
            source = ImageSource.NATIVE
        else:
            x0, y0, x1, y1 = (round(v) for v in bbox)
            data = crop_from_render(render_path, (x0, y0, x1, y1))
            filename = asset_name(page_number, index, "png")
            (assets_dir / filename).write_bytes(data)
    except ValueError, OSError:
        return None
    assert filename is not None
    await repo.record_image(
        session,
        job_id=job_id,
        page_number=page_number,
        asset_path=f"assets/{filename}",
        source=source,
        bbox={"bbox": list(bbox)},
        alt_text=shorten_alt(alt),
        caption=caption,
    )
    return filename, source


async def resolve_page_figures(
    session: AsyncSession,
    *,
    job_id: UUID,
    page_number: int,
    markdown: str,
    figures: list[FigurePayload],
    pdf_path: Path,
    renders_dir: Path,
    assets_dir: Path,
    fig_details: bool = True,
) -> ResolveResult:
    """Resolve one page's placeholders → asset links; records images rows.

    Native objects back a placeholder only when their placed geometry
    matches the agent box (overlap + comparable size); otherwise the
    figure is cropped from the page render. Count/order alignment is
    NEVER trusted — a lone icon on the page must not stand in for the
    flagged figure. Token page numbers are NOT trusted for matching
    (agents misnumber them): placement follows WHERE the token sits,
    so every token in the passed markdown resolves against this page. Token
    failures leave the token for QA (5.4). Payload entries with NO matching
    token (agent described a figure it never anchored) are APPENDED at page
    end — never silently dropped (NFR-1) — and reported in
    ResolveResult.orphaned for the report. Flushes rows; caller commits.
    """
    refs = parse_placeholders(markdown)
    payload_by_index = {fig.index: fig for fig in figures}
    if not refs and not payload_by_index:
        return ResolveResult(markdown=markdown)
    assets_dir.mkdir(parents=True, exist_ok=True)
    try:
        native = extract_native_images(pdf_path, page_number)
    except ValueError:
        native = []
    render_path = renders_dir / render_filename(page_number)
    native_match = match_natives_by_geometry(
        [ref.bbox for ref in refs], native, pdf_path, page_number, render_path
    )
    updated = markdown
    resolved = 0
    orphaned: list[dict[str, object]] = []
    matched_indexes: set[int] = set()
    for order, ref in enumerate(refs):
        payload = payload_by_index.get(ref.index)
        alt = payload.alt if payload is not None else f"Figure on page {page_number}"
        caption = payload.caption if payload is not None else None
        blob = None
        if payload is not None:
            matched_indexes.add(ref.index)
            matched = native_match.get(order)
            if matched is not None:
                blob = (native[matched].ext, native[matched].data)
        saved = await _save_figure(
            session,
            job_id=job_id,
            page_number=page_number,
            index=ref.index,
            bbox=ref.bbox,
            alt=alt,
            caption=caption,
            blob=blob,
            render_path=renders_dir / render_filename(page_number),
            assets_dir=assets_dir,
        )
        if saved is None:
            continue  # leave token: integrity check fails it loudly (5.4)
        filename, _ = saved
        updated = updated.replace(ref.token, figure_block(filename, alt, caption, fig_details), 1)
        resolved += 1
    for index, payload in payload_by_index.items():
        if index in matched_indexes:
            continue
        x0, y0, x1, y1 = (round(v) for v in payload.bbox)
        try:
            data = crop_from_render(render_path, (x0, y0, x1, y1))
        except ValueError, OSError:
            orphaned.append(
                {
                    "index": index,
                    "alt": payload.alt,
                    "placed": False,
                    "reason": "bbox outside render; needs_review",
                }
            )
            continue
        filename = asset_name(page_number, index, "png")
        (assets_dir / filename).write_bytes(data)
        await repo.record_image(
            session,
            job_id=job_id,
            page_number=page_number,
            asset_path=f"assets/{filename}",
            source=ImageSource.CROP,
            bbox={"bbox": list(payload.bbox)},
            alt_text=shorten_alt(payload.alt),
            caption=payload.caption,
        )
        updated = (
            updated.rstrip("\n")
            + "\n\n"
            + figure_block(filename, payload.alt, payload.caption, fig_details)
            + "\n"
        )
        resolved += 1
        orphaned.append(
            {
                "index": index,
                "alt": payload.alt,
                "placed": True,
                "reason": "no placeholder token in markdown; appended at page end",
            }
        )
    return ResolveResult(markdown=updated, resolved=resolved, orphaned=orphaned)


def find_fig_leftovers(markdown: str) -> list[str]:
    """Every leftover FIG token: well-formed plus malformed (e.g. literal idx).

    parse_placeholders() intentionally ignores malformed tokens for
    resolution, but integrity must catch them — a literal
    <!--FIG:page:idx:...--> is an unresolved figure, never clean output.
    Also catches an unterminated marker with no closing -->.
    """
    well_formed = [ref.token for ref in parse_placeholders(markdown)]
    any_tokens = FIG_ANY_PATTERN.findall(markdown)
    leftovers = list(well_formed)
    for token in any_tokens:
        if token not in leftovers:
            leftovers.append(token)
    if FIG_MARKER in markdown and not any_tokens:
        leftovers.append(f"{FIG_MARKER} (unterminated)")
    return leftovers


def check_assets(markdown: str, assets_dir: Path) -> AssetReport:
    """Bijection proof (5.4): links resolve, files referenced, no leftovers.

    A missing assets dir means zero files: links (if any) report missing,
    which fails loudly instead of raising FileNotFoundError.
    """
    linked = LINK_PATTERN.findall(markdown)
    if not assets_dir.is_dir():
        return AssetReport(
            ok=not linked and not find_fig_leftovers(markdown),
            missing_files=list(linked),
            leftover_placeholders=find_fig_leftovers(markdown),
        )
    missing = [link for link in linked if not (assets_dir / Path(link).name).is_file()]
    referenced = {Path(link).name for link in linked}
    orphans = sorted(
        path.name for path in assets_dir.iterdir() if path.is_file() and path.name not in referenced
    )
    leftovers = find_fig_leftovers(markdown)
    ok = not missing and not orphans and not leftovers
    return AssetReport(
        ok=ok, missing_files=missing, orphan_files=orphans, leftover_placeholders=leftovers
    )


__all__ = [
    "AssetReport",
    "FigRef",
    "FigurePayload",
    "ResolveResult",
    "cap_native_pixels",
    "check_assets",
    "figure_block",
    "find_fig_leftovers",
    "match_natives_by_geometry",
    "parse_placeholders",
    "resolve_page_figures",
    "shorten_alt",
]
