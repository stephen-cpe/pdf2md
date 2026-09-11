"""Unit tier: bijection proofs on tmp dirs (no DB)."""

import pymupdf

from src.pdf import extract_native_images, render_page
from src.pipeline.images import (
    AssetReport,
    _native_match_score,
    check_assets,
    match_natives_by_geometry,
    parse_placeholders,
)


def test_parse_placeholders(tmp_path) -> None:
    refs = parse_placeholders("a <!--FIG:2:3:0,0,100,50--> b <!--FIG:2:4:1,2,3,4-->")
    assert [(r.page, r.index, r.bbox) for r in refs] == [
        (2, 3, (0.0, 0.0, 100.0, 50.0)),
        (2, 4, (1.0, 2.0, 3.0, 4.0)),
    ]
    # SRS FR-AGT-3 long form with literal "page:" (what the prompt specifies).
    long_form = parse_placeholders("a <!--FIG:page:1:0,0,100,50--> b")
    assert [(r.page, r.index, r.bbox) for r in long_form] == [(None, 1, (0.0, 0.0, 100.0, 50.0))]
    assert parse_placeholders("no tokens") == []
    assert parse_placeholders("<!--FIG:bad-->") == []


def _write(assets, name: str, data: bytes = b"x") -> None:
    assets.mkdir(exist_ok=True)
    (assets / name).write_bytes(data)


def test_bijection_ok(tmp_path) -> None:
    assets = tmp_path / "assets"
    _write(assets, "page-001-img-01.png")
    report = check_assets("![a](assets/page-001-img-01.png)", assets)
    assert report == AssetReport(
        ok=True, missing_files=[], orphan_files=[], leftover_placeholders=[]
    )


def test_dangling_link(tmp_path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    report = check_assets("![a](assets/gone.png)", assets)
    assert not report.ok and report.missing_files == ["assets/gone.png"]


def test_orphan_file(tmp_path) -> None:
    assets = tmp_path / "assets"
    _write(assets, "page-001-img-01.png")
    _write(assets, "page-001-img-02.png")
    report = check_assets("![a](assets/page-001-img-01.png)", assets)
    assert not report.ok and report.orphan_files == ["page-001-img-02.png"]


def test_leftover_placeholder(tmp_path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    report = check_assets("<!--FIG:1:1:0,0,10,10-->", assets)
    assert not report.ok and report.leftover_placeholders == ["<!--FIG:1:1:0,0,10,10-->"]


def test_literal_idx_placeholder_fails_loudly(tmp_path) -> None:
    """Agent copying the prompt's INDEX placeholder verbatim must hard-fail."""
    from src.pipeline.images import find_fig_leftovers

    assets = tmp_path / "assets"
    assets.mkdir()
    bad = "<!--FIG:page:idx:510,1400,1190,1590-->"
    # Resolution ignores it (no numeric index to pair), ...
    assert parse_placeholders(bad) == []
    # ... but integrity catches it.
    assert find_fig_leftovers(bad) == [bad]
    report = check_assets(bad, assets)
    assert not report.ok and report.leftover_placeholders == [bad]
    mixed = check_assets(f"ok text\n{bad}\n![a](assets/x.png)", assets)
    assert not mixed.ok and bad in mixed.leftover_placeholders


def test_unterminated_fig_marker_fails(tmp_path) -> None:
    from src.pipeline.images import find_fig_leftovers

    assets = tmp_path / "assets"
    assets.mkdir()
    assert find_fig_leftovers("text <!--FIG:page:1:0,0,10") != []
    assert not check_assets("text <!--FIG:page:1:0,0,10", assets).ok


def test_native_match_score_gates() -> None:
    # Same box matches with IoU 1.0.
    assert _native_match_score((0.0, 0.0, 100.0, 100.0), (0.0, 0.0, 100.0, 100.0)) == 1.0
    # Disjoint boxes never pair.
    assert _native_match_score((0.0, 0.0, 100.0, 100.0), (200.0, 200.0, 300.0, 300.0)) is None
    # Degenerate boxes never pair.
    assert _native_match_score((0.0, 0.0, 0.0, 100.0), (0.0, 0.0, 100.0, 100.0)) is None
    # Tiny icon fully inside a big agent box: containment passes but the
    # area ratio (0.3%) rejects it — arxiv p4 badge vs flowchart.
    assert _native_match_score((0.0, 0.0, 960.0, 370.0), (100.0, 100.0, 134.0, 134.0)) is None
    # Page-spanning union vs one figure: containment passes but 16x area rejects it.
    assert _native_match_score((100.0, 100.0, 200.0, 200.0), (0.0, 0.0, 400.0, 400.0)) is None
    # Half overlap at the same size still pairs (agent estimates are rough).
    assert _native_match_score((0.0, 0.0, 100.0, 100.0), (50.0, 0.0, 150.0, 100.0)) == (
        5000.0 / 15000.0
    )


def _icon_pdf(path) -> bytes:
    """400x400pt page with one small JPEG icon at (10,10,60,60) + its bytes."""
    render_doc = pymupdf.open()
    render_doc.new_page(width=200, height=100).insert_text((10, 50), "native pixels")
    pix = render_doc[0].get_pixmap(matrix=pymupdf.Matrix(1, 1))
    jpeg = bytes(pix.tobytes("jpg"))
    render_doc.close()
    doc = pymupdf.open()
    doc.new_page(width=400, height=400).insert_image(pymupdf.Rect(10, 10, 60, 60), stream=jpeg)
    doc.save(path)
    doc.close()
    return jpeg


def test_match_natives_by_geometry(tmp_path) -> None:
    pdf = tmp_path / "doc.pdf"
    _icon_pdf(pdf)
    natives = extract_native_images(pdf, 1)
    assert len(natives) == 1 and natives[0].bbox is not None
    renders = tmp_path / "r"
    render_page(pdf, 1, 200, renders / "page-001.png")  # 1111x1111px
    render_path = renders / "page-001.png"
    # Big agent box swallowing the icon: rejected by area ratio → crop.
    assert match_natives_by_geometry([(0.0, 0.0, 600.0, 320.0)], natives, pdf, 1, render_path) == {}
    # Tight box around the icon placement (27,27,167,167 render px): pairs.
    assert match_natives_by_geometry(
        [(20.0, 20.0, 170.0, 170.0)], natives, pdf, 1, render_path
    ) == {0: 0}
    # Far-away box: no overlap → crop.
    assert (
        match_natives_by_geometry([(800.0, 800.0, 1000.0, 1000.0)], natives, pdf, 1, render_path)
        == {}
    )
    # Missing render falls back to assumed DPI instead of crashing.
    assert match_natives_by_geometry(
        [(20.0, 20.0, 170.0, 170.0)], natives, pdf, 1, renders / "absent.png"
    ) == {0: 0}


def _pil_bytes(width: int, height: int, fmt: str) -> bytes:
    from io import BytesIO

    from PIL import Image

    img = Image.new("RGB", (width, height), (10, 128, 200))
    out = BytesIO()
    img.save(out, format=fmt)
    return out.getvalue()


def _pil_size(data: bytes) -> tuple[int, int]:
    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(data)) as img:
        return img.size


def test_cap_native_pixels() -> None:
    from src.pipeline.images import MAX_NATIVE_DIM, cap_native_pixels

    small_jpg = _pil_bytes(200, 100, "JPEG")
    assert cap_native_pixels(small_jpg, "jpg") == ("jpg", small_jpg)  # verbatim passthrough
    exact = _pil_bytes(MAX_NATIVE_DIM, 400, "PNG")
    assert cap_native_pixels(exact, "png") == ("png", exact)
    big_png = _pil_bytes(3000, 2000, "PNG")
    ext, capped = cap_native_pixels(big_png, "png")
    assert ext == "png" and capped != big_png
    assert _pil_size(capped) == (1600, 1067)  # aspect preserved, long edge capped
    big_jpg = _pil_bytes(3886, 3885, "JPEG")
    ext, capped_jpg = cap_native_pixels(big_jpg, "jpg")
    assert ext == "jpg"
    assert max(_pil_size(capped_jpg)) <= MAX_NATIVE_DIM
    assert cap_native_pixels(b"not an image", "png") == ("png", b"not an image")


def test_missing_dir_means_no_files_not_crash(tmp_path) -> None:
    # M5 bug: integrity on a figureless job crashed with FileNotFoundError
    # because the assets dir was never created. Now: clean bill when empty.
    missing = tmp_path / "no-assets"
    assert check_assets("plain text, no figures", missing).ok
    report = check_assets("![a](assets/gone.png)", missing)
    assert not report.ok and report.missing_files == ["assets/gone.png"]
