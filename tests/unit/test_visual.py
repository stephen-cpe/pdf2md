"""Unit tier: triplet artifacts + diff sanity (tmp PNGs)."""

import pymupdf

from src.pipeline.visual import diagnostic_triplet, diff_images, render_text_png


def _png(path, color: tuple[int, int, int], width: int = 100, height: int = 80) -> None:
    doc = pymupdf.open()
    try:
        page = doc.new_page(width=width, height=height)
        page.draw_rect(
            page.rect, color=tuple(c / 255 for c in color), fill=tuple(c / 255 for c in color)
        )
        page.get_pixmap().save(str(path))
    finally:
        doc.close()


def test_identical_images_diff_zero(tmp_path) -> None:
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    _png(first, (10, 20, 30))
    _png(second, (10, 20, 30))
    assert diff_images(first, second, tmp_path / "d.png") == 0.0


def test_different_images_diff_positive(tmp_path) -> None:
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    _png(first, (0, 0, 0))
    _png(second, (255, 255, 255))
    assert diff_images(first, second, tmp_path / "d.png") > 90.0


def test_triplet_artifacts(tmp_path) -> None:
    source = tmp_path / "source.png"
    _png(source, (200, 200, 200))
    debug = tmp_path / "debug"
    diagnostic = diagnostic_triplet(source, "# Hello\n\nbody text", debug, 7)
    assert diagnostic.page_number == 7
    assert diagnostic.source_path.is_file()
    assert diagnostic.rendered_path.is_file()
    assert diagnostic.diff_path.is_file()
    assert diagnostic.diff_pct >= 0.0
    assert render_text_png("x", 50, 40, tmp_path / "r.png").is_file()
