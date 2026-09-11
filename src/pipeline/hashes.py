"""Deterministic provenance hashes.

source_page_hash pins the SOURCE (PDF content streams of that page):
same file → same hash, any byte change → new hash. render/ocr/markdown
hashes pin each pipeline artifact. All SHA-256 hex.
"""

import hashlib
from pathlib import Path

import pymupdf


def sha256_hex(data: bytes) -> str:
    """Hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def text_hash(text: str) -> str:
    """Hex digest of UTF-8 text."""
    return sha256_hex(text.encode("utf-8"))


def file_hash(path: Path) -> str:
    """Hex digest of a file (streamed)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def page_source_hash(pdf_path: Path, page_number: int) -> str:
    """Hex digest of one page's PDF content streams + mediabox (1-based)."""
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_number - 1]
        digest = hashlib.sha256()
        for xref in page.get_contents():
            digest.update(doc.xref_stream(xref))
        digest.update(f"{page.rect}".encode())
        return digest.hexdigest()
    finally:
        doc.close()


__all__ = ["file_hash", "page_source_hash", "sha256_hex", "text_hash"]
