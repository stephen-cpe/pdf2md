"""fixtures valid/corrupt/encrypted/caps; render; both extract paths."""

import math
from pathlib import Path

import pymupdf
import pytest

from src.pdf import (
    asset_name,
    crop_from_render,
    extract_native_images,
    native_text,
    native_text_if_viable,
    native_text_in_bbox,
    preflight,
    render_filename,
    render_page,
)


def _text_pdf(path: Path, pages: int = 2) -> Path:
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), "Hello deterministic world. " * 10)
    doc.set_metadata({"title": "Fixture Doc"})
    doc.save(path)
    doc.close()
    return path


def _scanned_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(10, 10, 200, 200))
    doc.save(path)
    doc.close()
    return path


def _corrupt_file(path: Path) -> Path:
    path.write_bytes(b"\x00\x01not a pdf at all\xff\xfe" * 64)
    return path


def _encrypted_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "secret text here")
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    doc.close()
    return path


def _image_pdf(path: Path) -> tuple[Path, bytes]:
    # Real JPEG bytes, self-generated (render → jpg), then embedded as-is.
    render_doc = pymupdf.open()
    render_doc.new_page(width=200, height=100).insert_text((10, 50), "jpeg pixels")
    pix = render_doc[0].get_pixmap(matrix=pymupdf.Matrix(1, 1))
    jpeg = bytes(pix.tobytes("jpg"))
    png = bytes(pix.tobytes("png"))
    render_doc.close()
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    page.insert_image(pymupdf.Rect(10, 10, 210, 110), stream=jpeg)
    page.insert_image(pymupdf.Rect(10, 120, 210, 220), stream=png)
    doc.save(path)
    doc.close()
    return path, jpeg


# --- 2.1 preflight ---


def test_preflight_valid(tmp_path) -> None:
    pdf = _text_pdf(tmp_path / "valid.pdf")
    result = preflight(pdf)
    assert result.valid and result.page_count == 2
    assert result.metadata.get("title") == "Fixture Doc"
    assert result.born_digital
    assert result.width_pts > 0 and result.height_pts > 0


def test_preflight_scanned(tmp_path) -> None:
    result = preflight(_scanned_pdf(tmp_path / "scan.pdf"))
    assert result.valid and not result.born_digital


def test_preflight_missing() -> None:
    result = preflight(Path("does-not-exist.pdf"))
    assert not result.valid and "not found" in result.error


def test_preflight_corrupt(tmp_path) -> None:
    result = preflight(_corrupt_file(tmp_path / "bad.pdf"))
    assert not result.valid and "corrupt" in result.error


def test_preflight_encrypted(tmp_path) -> None:
    result = preflight(_encrypted_pdf(tmp_path / "enc.pdf"))
    assert not result.valid and "password" in result.error


def test_preflight_caps(tmp_path) -> None:
    pdf = _text_pdf(tmp_path / "valid.pdf")
    assert not preflight(pdf, max_pages=1).valid
    assert "cap" in preflight(pdf, max_pages=1).error
    assert not preflight(pdf, max_mb=1e-6).valid


# --- 2.2 render ---


def test_render_exact_dims_and_names(tmp_path) -> None:
    pdf = _text_pdf(tmp_path / "valid.pdf", pages=1)
    rect = pymupdf.open(pdf)[0].rect
    out = tmp_path / render_filename(1)
    info = render_page(pdf, 1, 200, out)
    assert out.exists()
    assert (info.width, info.height) == (
        round(rect.width * 200 / 72),
        round(rect.height * 200 / 72),
    )
    assert info.page_count == 1


def test_render_bump_and_limits(tmp_path) -> None:
    pdf = _text_pdf(tmp_path / "valid.pdf", pages=1)
    info = render_page(pdf, 1, 300, tmp_path / "hi.png")
    # MuPDF rounds fractional pixels up.
    assert (info.width, info.height) == (math.ceil(595 * 300 / 72), math.ceil(842 * 300 / 72))
    with pytest.raises(ValueError):
        render_page(pdf, 1, 500, tmp_path / "x.png")
    with pytest.raises(ValueError):
        render_page(pdf, 0, 200, tmp_path / "x.png")
    with pytest.raises(ValueError):
        render_page(pdf, 9, 200, tmp_path / "x.png")


# --- 2.3 extract ---


def test_extract_native_jpg_passthrough_and_png(tmp_path) -> None:
    pdf, jpeg = _image_pdf(tmp_path / "img.pdf")
    found = extract_native_images(pdf, 1)
    assert len(found) == 2
    by_ext = {img.ext for img in found}
    assert by_ext == {"jpg", "png"}
    passthrough = next(img for img in found if img.ext == "jpg")
    assert passthrough.data == jpeg
    assert all(img.source == "native" for img in found)


def test_crop_from_render(tmp_path) -> None:
    pdf = _text_pdf(tmp_path / "valid.pdf", pages=1)
    render = tmp_path / "page.png"
    render_page(pdf, 1, 200, render)
    data = crop_from_render(render, (10, 10, 110, 60))
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pix = pymupdf.Pixmap(data)
    assert (pix.width, pix.height) == (100, 50)
    with pytest.raises(ValueError):
        crop_from_render(render, (50, 50, 10, 10))


def test_asset_name() -> None:
    assert asset_name(1, 1, "png") == "page-001-img-01.png"
    assert asset_name(42, 3, "jpg") == "page-042-img-03.jpg"


# --- native_text_in_bbox (diagram grounding) ---


def _two_region_pdf(path: Path) -> Path:
    """One page, top label vs bottom label in separate regions."""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    page.insert_text((50, 60), "TOPLABEL", fontsize=12)
    page.insert_text((50, 360), "BOTTOMLABEL", fontsize=12)
    doc.save(path)
    doc.close()
    return path


def test_native_text_in_bbox_returns_only_region_text(tmp_path) -> None:
    pdf = _two_region_pdf(tmp_path / "regions.pdf")
    # No render: bbox is interpreted as PDF points 1:1.
    bottom = native_text_in_bbox(pdf, 1, (0.0, 300.0, 400.0, 400.0))
    top = native_text_in_bbox(pdf, 1, (0.0, 0.0, 400.0, 100.0))
    assert "BOTTOMLABEL" in bottom and "TOPLABEL" not in bottom
    assert "TOPLABEL" in top and "BOTTOMLABEL" not in top


def test_native_text_in_bbox_scales_render_pixels(tmp_path) -> None:
    """Agent bboxes are render pixels; the helper converts via render dims."""
    pdf = _two_region_pdf(tmp_path / "regions.pdf")
    render = tmp_path / "page.png"
    render_page(pdf, 1, 144, render)  # 2x scale (144/72)
    # Bottom region in rendered pixels (200..800 y at 2x).
    bottom = native_text_in_bbox(pdf, 1, (0.0, 600.0, 800.0, 800.0), render)
    top = native_text_in_bbox(pdf, 1, (0.0, 0.0, 800.0, 200.0), render)
    assert "BOTTOMLABEL" in bottom and "TOPLABEL" not in bottom
    assert "TOPLABEL" in top and "BOTTOMLABEL" not in top


def test_native_text_in_bbox_empty_and_defective(tmp_path) -> None:
    pdf = _two_region_pdf(tmp_path / "regions.pdf")
    assert native_text_in_bbox(pdf, 1, (0.0, 150.0, 400.0, 250.0)).strip() == ""
    assert native_text_in_bbox(pdf, 9, (0.0, 0.0, 10.0, 10.0)) == ""
    assert native_text_in_bbox(tmp_path / "missing.pdf", 1, (0.0, 0.0, 10.0, 10.0)) == ""


def test_native_text_still_returns_whole_page(tmp_path) -> None:
    pdf = _two_region_pdf(tmp_path / "regions.pdf")
    whole = native_text(pdf, 1)
    assert "TOPLABEL" in whole and "BOTTOMLABEL" in whole


# --- native_text_if_viable (per-page reference routing) ---


def test_native_text_if_viable_substantial(tmp_path) -> None:
    pdf = _text_pdf(tmp_path / "valid.pdf", pages=1)
    text = native_text_if_viable(pdf, 1, min_words=20)
    assert text is not None and "Hello deterministic" in text


def test_native_text_if_viable_sparse_returns_none(tmp_path) -> None:
    pdf = _two_region_pdf(tmp_path / "regions.pdf")  # 2 words on the page
    assert native_text_if_viable(pdf, 1, min_words=20) is None


def test_native_text_if_viable_prose_below_words_returns_none(tmp_path) -> None:
    """Ordinary prose with few words must not trip the CJK char fallback."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "driver figure page " * 10)  # 12 words, 72 chars
    doc.save(tmp_path / "prose.pdf")
    doc.close()
    assert len(native_text(tmp_path / "prose.pdf", 1).split()) < 20
    assert native_text_if_viable(tmp_path / "prose.pdf", 1, min_words=20) is None


def test_native_text_if_viable_scanned_returns_none(tmp_path) -> None:
    pdf = _scanned_pdf(tmp_path / "scan.pdf")
    assert native_text_if_viable(pdf, 1, min_words=20) is None


def test_native_text_if_viable_cjk_char_fallback(tmp_path) -> None:
    """Space-less scripts count as one word; the char fallback keeps them.

    PyMuPDF clips the inserted CJK run to one line (~48 chars), which is
    under the word threshold but over the 3-chars-per-word fallback.
    """
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72), "\u4e2d\u6587\u6d4b\u8bd5\u5185\u5bb9\u793a\u4f8b" * 10, fontname="china-s"
    )
    doc.save(tmp_path / "cjk.pdf")
    doc.close()
    text = native_text_if_viable(tmp_path / "cjk.pdf", 1, min_words=10)
    assert text is not None and len(text.strip()) >= 30


def test_native_text_if_viable_defective_returns_none(tmp_path) -> None:
    assert native_text_if_viable(tmp_path / "missing.pdf", 1, min_words=20) is None
