"""Hash determinism + sensitivity."""

import pymupdf

from src.pipeline.hashes import file_hash, page_source_hash, text_hash


def _pdf(path, pages: int = 2):
    from pathlib import Path

    doc = pymupdf.open()
    for number in range(pages):
        doc.new_page().insert_text((72, 72), f"content {number}")
    doc.save(Path(path))
    doc.close()
    return Path(path)


def test_hashes_deterministic(tmp_path) -> None:
    pdf = _pdf(tmp_path / "a.pdf")
    assert page_source_hash(pdf, 1) == page_source_hash(pdf, 1)
    assert text_hash("hello") == text_hash("hello")
    assert file_hash(pdf) == file_hash(pdf)


def test_hashes_sensitive(tmp_path) -> None:
    pdf = _pdf(tmp_path / "a.pdf")
    assert page_source_hash(pdf, 1) != page_source_hash(pdf, 2)
    assert text_hash("hello") != text_hash("goodbye")
    other = _pdf(tmp_path / "b.pdf")
    assert file_hash(pdf) != file_hash(other)
