"""Hybrid routing — decide_route/extract_deterministic semantics.

Unit tier: synthetic PDFs, no DB/Ollama/Cloud.
"""

from pathlib import Path

import pymupdf
import pytest

from src.pipeline.routing import (
    MAX_DECORATIVE_IMAGE_AREA,
    decide_route,
    extract_deterministic,
    native_text,
    route_page,
)


def _text_pdf(path: Path, pages: int = 2, words_per_page: int = 40) -> Path:
    """Born-digital PDF: every page has a rich native text layer.

    Zero-padded per-page tokens (word005) make page-content assertions
    unambiguous (word15 can never appear inside word150).
    """
    doc = pymupdf.open()
    for number in range(pages):
        page = doc.new_page()
        body = " ".join(f"word{number:02d}{i:03d}" for i in range(words_per_page))
        page.insert_textbox(pymupdf.Rect(72, 72, 540, 720), body, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def _scanned_pdf(path: Path) -> Path:
    """Image-only page: no native text layer at all."""
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100))
    doc.new_page().insert_image(pymupdf.Rect(0, 0, 612, 792), pixmap=pix)
    doc.save(path)
    doc.close()
    return path


def _figure_pdf(path: Path) -> Path:
    """Born-digital page WITH a large placed image (content figure)."""
    doc = pymupdf.open()
    page = doc.new_page()
    body = " ".join(f"figword{i:03d}" for i in range(40))
    page.insert_textbox(pymupdf.Rect(72, 72, 540, 180), body, fontsize=11)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100))
    # image covers ~30% of the page: clearly a content figure
    page.insert_image(pymupdf.Rect(72, 200, 400, 450), pixmap=pix)
    doc.save(path)
    doc.close()
    return path


def test_native_text_extracted(tmp_path: Path) -> None:
    pdf = _text_pdf(tmp_path / "t.pdf")
    text = native_text(pdf, 1)
    assert "word00005" in text


def test_native_text_page_number_is_1_based(tmp_path: Path) -> None:
    pdf = _text_pdf(tmp_path / "t.pdf", pages=2)
    assert "word00000" in native_text(pdf, 1) and "word01000" not in native_text(pdf, 1)
    assert "word01000" in native_text(pdf, 2)


def test_native_text_out_of_range_raises(tmp_path: Path) -> None:
    pdf = _text_pdf(tmp_path / "t.pdf", pages=1)
    with pytest.raises(ValueError):
        native_text(pdf, 2)


def test_born_digital_text_page_routes_deterministic(tmp_path: Path) -> None:
    pdf = _text_pdf(tmp_path / "t.pdf")
    decision = decide_route(pdf, 1)
    assert decision.deterministic and decision.native_words >= 20
    assert "native text present" in decision.reason


def test_scanned_page_routes_agentic(tmp_path: Path) -> None:
    pdf = _scanned_pdf(tmp_path / "s.pdf")
    decision = decide_route(pdf, 1)
    assert not decision.deterministic and "sparse" in decision.reason


def test_figure_page_routes_agentic(tmp_path: Path) -> None:
    """Large placed image = content figure → agentic (deterministic
    extraction drops image data; figures must be resolved)."""
    pdf = _figure_pdf(tmp_path / "f.pdf")
    decision = decide_route(pdf, 1)
    assert not decision.deterministic and "image area" in decision.reason


def test_extract_deterministic_returns_markdown(tmp_path: Path) -> None:
    pdf = _text_pdf(tmp_path / "t.pdf")
    decision, candidate = route_page(pdf, 1)
    assert decision.deterministic
    assert candidate is not None and "word00005" in candidate


def test_extract_deterministic_page_scoped(tmp_path: Path) -> None:
    """pages=[0] (0-based physical page 1) must not leak page 2 content."""
    pdf = _text_pdf(tmp_path / "t.pdf", pages=3, words_per_page=60)
    _, candidate = route_page(pdf, 1)
    assert candidate is not None
    assert "word00015" in candidate and "word01015" not in candidate


def test_route_page_scanned_returns_none_candidate(tmp_path: Path) -> None:
    pdf = _scanned_pdf(tmp_path / "s.pdf")
    decision, candidate = route_page(pdf, 1)
    assert not decision.deterministic and candidate is None


def test_extract_deterministic_survives_odd_pdf(tmp_path: Path) -> None:
    """Corrupt/undecodable page: extraction returns None (escalates),
    never raises — routing must not crash the pipeline."""
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")
    assert extract_deterministic(bad, 1) is None
    decision = decide_route(bad, 1)
    assert not decision.deterministic  # native_text raises → sparse


def test_decorative_cap_constant_is_sane() -> None:
    assert 0 < MAX_DECORATIVE_IMAGE_AREA < 0.5
