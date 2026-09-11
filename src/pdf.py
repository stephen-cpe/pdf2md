"""Deterministic PDF processing — preflight, render, image extract.

No LLMs here. Page numbers are 1-based throughout (matches the repository
layer, UI "page N of M", and page-{n:03d} asset names).
"""

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

MAX_RENDER_DPI = 400
_TEXT_SAMPLE_PAGES = 3
_TEXT_SAMPLE_MIN_CHARS = 50


@dataclass(frozen=True)
class PreflightResult:
    """Preflight verdict (FR-PDF-1/2)."""

    valid: bool
    page_count: int = 0
    metadata: dict[str, str] = field(default_factory=dict)
    width_pts: float = 0.0
    height_pts: float = 0.0
    born_digital: bool = False
    error: str = ""


def preflight(pdf_path: Path, max_mb: float = 500.0, max_pages: int = 1000) -> PreflightResult:
    """Validate + classify one PDF (FR-PDF-1/2, FR-JOB-1 caps)."""
    path = Path(pdf_path)
    if not path.is_file():
        return PreflightResult(valid=False, error=f"not found: {path}")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        return PreflightResult(valid=False, error=f"size {size_mb:.1f} MB exceeds cap {max_mb} MB")
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        return PreflightResult(valid=False, error=f"unreadable or corrupt PDF: {exc}")
    try:
        if doc.is_encrypted:
            return PreflightResult(valid=False, error="password-protected PDF (encrypted)")
        count = doc.page_count
        if count > max_pages:
            return PreflightResult(valid=False, error=f"page count {count} exceeds cap {max_pages}")
        if count == 0:
            return PreflightResult(valid=False, error="PDF has no pages")
        rect = doc[0].rect
        meta = {k: str(v) for k, v in (doc.metadata or {}).items()}
        born = any(
            len(doc[i].get_text().strip()) >= _TEXT_SAMPLE_MIN_CHARS
            for i in range(min(_TEXT_SAMPLE_PAGES, count))
        )
        return PreflightResult(
            valid=True,
            page_count=count,
            metadata=meta,
            width_pts=rect.width,
            height_pts=rect.height,
            born_digital=born,
        )
    finally:
        doc.close()


@dataclass(frozen=True)
class RenderInfo:
    """One rendered page (FR-PDF-3/4)."""

    width: int
    height: int
    dpi: int
    page_count: int


def render_filename(page_number: int) -> str:
    """Deterministic render name (re-renders overwrite the same file)."""
    return f"page-{page_number:03d}.png"


def render_page(pdf_path: Path, page_number: int, dpi: int, output_path: Path) -> RenderInfo:
    """Render a 1-based page to PNG (FR-PDF-3; DPI bump to 400 max, FR-PDF-4)."""
    if dpi > MAX_RENDER_DPI:
        raise ValueError(f"dpi {dpi} exceeds max {MAX_RENDER_DPI}")
    if page_number < 1:
        raise ValueError(f"page_number is 1-based, got {page_number}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72.0
    doc = pymupdf.open(pdf_path)
    try:
        count = doc.page_count
        if page_number > count:
            raise ValueError(f"page {page_number} out of range (1..{count})")
        pix = doc[page_number - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        pix.save(str(output_path))
        return RenderInfo(width=pix.width, height=pix.height, dpi=dpi, page_count=count)
    finally:
        doc.close()


@dataclass(frozen=True)
class ExtractedImage:
    """One native figure (FR-IMG-1/2): bytes + format + provenance."""

    data: bytes
    ext: str  # "jpg" passthrough, else "png"
    source: str  # always "native" here; "crop" comes from crop_from_render
    bbox: tuple[float, float, float, float] | None = (
        None  # PDF-point placement (union of occurrences)
    )


def asset_name(page_number: int, index: int, ext: str) -> str:
    """Deterministic asset name (FR-IMG-2): page-{p:03d}-img-{n:02d}.{ext}."""
    return f"page-{page_number:03d}-img-{index:02d}.{ext}"


def extract_native_images(pdf_path: Path, page_number: int) -> list[ExtractedImage]:
    """Native PDF image objects first (FR-IMG-1); masks excluded."""
    if page_number < 1:
        raise ValueError(f"page_number is 1-based, got {page_number}")
    doc = pymupdf.open(pdf_path)
    try:
        if page_number > doc.page_count:
            raise ValueError(f"page {page_number} out of range (1..{doc.page_count})")
        page = doc[page_number - 1]
        entries = page.get_images(full=True)
        masks = {e[1] for e in entries if len(e) > 1 and e[1] != 0}
        found: list[ExtractedImage] = []
        seen: set[int] = set()
        for entry in entries:
            xref = int(entry[0])
            if xref in seen or xref in masks:
                continue
            seen.add(xref)
            try:
                # Full entry tuple, not the bare xref (get_image_bbox rejects ints).
                rect = page.get_image_bbox(entry)
                placement: tuple[float, float, float, float] | None = (
                    rect.x0,
                    rect.y0,
                    rect.x1,
                    rect.y1,
                )
            except Exception:
                placement = None  # unplaced raster: geometry matcher skips it (crop instead)
            info = doc.extract_image(xref)
            ext = str(info["ext"]).lower()
            if ext in ("jpg", "jpeg"):
                found.append(
                    ExtractedImage(
                        data=bytes(info["image"]), ext="jpg", source="native", bbox=placement
                    )
                )
                continue
            pix = pymupdf.Pixmap(doc, xref)
            if pix.n > 4:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            found.append(
                ExtractedImage(
                    data=bytes(pix.tobytes("png")), ext="png", source="native", bbox=placement
                )
            )
        return found
    finally:
        doc.close()


def crop_from_render(render_png: Path, bbox: tuple[int, int, int, int]) -> bytes:
    """Crop fallback (FR-IMG-1): absolute-pixel bbox from the rendered PNG.

    Implemented as a one-page PDF round-trip (stable APIs only): the PNG is
    placed 1:1 on a same-sized page, the page is cropboxed to the bbox, and
    re-rendered at zoom 1 — output pixels equal bbox pixels exactly.
    """
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 and 0 <= y0 < y1):
        raise ValueError(f"invalid bbox {bbox!r}")
    probe = pymupdf.Pixmap(str(render_png))
    width, height = probe.width, probe.height
    if not (x1 <= width and y1 <= height):
        raise ValueError(f"bbox {bbox!r} outside image {width}x{height}")
    doc = pymupdf.open()
    try:
        page = doc.new_page(width=width, height=height)
        page.insert_image(page.rect, filename=str(render_png))
        page.set_cropbox(pymupdf.Rect(x0, y0, x1, y1))
        return bytes(page.get_pixmap().tobytes("png"))
    finally:
        doc.close()


__all__ = [
    "ExtractedImage",
    "PreflightResult",
    "RenderInfo",
    "asset_name",
    "crop_from_render",
    "extract_native_images",
    "preflight",
    "render_filename",
    "render_page",
]
