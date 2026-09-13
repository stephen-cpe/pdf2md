"""Synthesize a scanned (image-only) corpus fixture for the OCR ablation.

Takes the first N pages of a born-digital source PDF, renders each page to a
PNG, and rebuilds a new PDF containing only those rasters — zero native text
layer. The vision/OCR stages then do all the work, which is exactly the
condition the no-OCR arm must prove itself under.

Usage:
    venv\\Scripts\\python tools\\make_scanned_fixture.py
"""

import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "corpus" / "Liskov_Substitution_Principle.pdf"
DEST = ROOT / "corpus" / "scanned_fixture.pdf"
PAGES = 3
DPI = 200


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f"source PDF not found: {SOURCE}")
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    src = pymupdf.open(SOURCE)
    try:
        count = min(PAGES, src.page_count)
        out = pymupdf.open()
        try:
            for number in range(count):
                page = src[number]
                pix = page.get_pixmap(matrix=pymupdf.Matrix(DPI / 72.0, DPI / 72.0))
                png = bytes(pix.tobytes("png"))
                fresh = out.new_page(width=page.rect.width, height=page.rect.height)
                fresh.insert_image(page.rect, stream=png)
            out.save(DEST)
        finally:
            out.close()
    finally:
        src.close()
    check = pymupdf.open(DEST)
    try:
        words = sum(len(check[i].get_text().split()) for i in range(check.page_count))
    finally:
        check.close()
    print(f"wrote {DEST} ({count} pages, native words: {words})")
    assert words == 0, "fixture must have no native text layer"


if __name__ == "__main__":
    main()
