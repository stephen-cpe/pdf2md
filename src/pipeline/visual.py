"""Visual fidelity diagnostic — developer-only, never user-shipped (NFR-13).

Renders sampled final Markdown back to a same-dimension page image (plain
text layout — a layout HEURISTIC, not a browser) and diffs it against the
source page render. Artifacts: debug/page-NNN-{source,rendered,diff}.png
plus a mean-abs-diff percentage. A high diff_pct on a text-faithful page
points at layout drift worth human eyes; it is a signal, not a verdict.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class VisualDiagnostic:
    """One sampled page's diagnostic triplet."""

    page_number: int
    source_path: Path
    rendered_path: Path
    diff_path: Path
    diff_pct: float


def render_text_png(markdown: str, width: int, height: int, out_path: Path) -> Path:
    """Render markdown as plain text onto a width×height page image."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    try:
        page = doc.new_page(width=width, height=height)
        inset = max(0, min(36, width // 2 - 1, height // 2 - 1))
        page.insert_textbox(
            pymupdf.Rect(inset, inset, width - inset, height - inset), markdown, fontsize=11
        )
        pix = page.get_pixmap()
        pix.save(str(out_path))
        return out_path
    finally:
        doc.close()


def _gray(pix: pymupdf.Pixmap) -> bytes:
    """Luminosity grayscale samples for one pixmap."""
    gray = pymupdf.Pixmap(pymupdf.csGRAY, pix)
    return bytes(gray.samples)


def diff_images(source_png: Path, rendered_png: Path, diff_path: Path) -> float:
    """Mean absolute grayscale diff percentage; writes a diff visualization."""
    src = pymupdf.Pixmap(str(source_png))
    ren = pymupdf.Pixmap(str(rendered_png))
    if (src.width, src.height) != (ren.width, ren.height):
        ren = pymupdf.Pixmap(ren, src.width, src.height)
    left, right = _gray(src), _gray(ren)
    total = sum(abs(a - b) for a, b in zip(left, right))
    pct = total / len(left) / 255.0 * 100.0 if left else 100.0
    diff_bytes = bytes(abs(a - b) for a, b in zip(left, right))
    diff = pymupdf.Pixmap(pymupdf.csGRAY, src.width, src.height, diff_bytes, 0)
    diff.save(str(diff_path))
    return pct


def diagnostic_triplet(
    source_png: Path, markdown_text: str, out_dir: Path, page_number: int
) -> VisualDiagnostic:
    """Full triplet for one sampled page (6.7 DoD unit)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path = out_dir / f"page-{page_number:03d}-source.png"
    rendered_path = out_dir / f"page-{page_number:03d}-rendered.png"
    diff_path = out_dir / f"page-{page_number:03d}-diff.png"
    shutil.copy2(source_png, source_path)
    probe = pymupdf.Pixmap(str(source_png))
    render_text_png(markdown_text, probe.width, probe.height, rendered_path)
    pct = diff_images(source_path, rendered_path, diff_path)
    return VisualDiagnostic(
        page_number=page_number,
        source_path=source_path,
        rendered_path=rendered_path,
        diff_path=diff_path,
        diff_pct=pct,
    )


__all__ = ["VisualDiagnostic", "diagnostic_triplet", "diff_images", "render_text_png"]
