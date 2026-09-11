"""Unit tier: merges, TOC, heading norm, dedup logic."""

from src.pipeline.assemble import (
    assemble,
    build_toc,
    dedup_furniture,
    join_pages,
    normalize_headings,
)


def _same(left: str, right: str) -> bool:
    return left.strip() == right.strip()


def test_table_merge_drops_dup_separator() -> None:
    page1 = "| a | b |\n|---|---|\n| 1 | 2 |"
    page2 = "|---|---| \n| 3 | 4 |"
    merged = join_pages([page1, page2])
    assert merged.count("|---|---|") == 1
    assert "| 1 | 2 |\n| 3 | 4 |" in merged


def test_list_continuation_stays_tight() -> None:
    merged = join_pages(["- one\n- two", "- three"])
    assert "- two\n- three" in merged


def test_normal_pages_separated() -> None:
    assert join_pages(["para one", "para two"]) == "para one\n\npara two\n"


def test_heading_normalize_shifts_to_h1() -> None:
    assert normalize_headings("## A\n### B").startswith("# A\n## B")
    assert normalize_headings("# A\n## B") == "# A\n## B"


def test_toc_anchors_github_style() -> None:
    toc = build_toc("# Hello, World!\n## What's New?")
    assert "- [Hello, World!](#hello-world)" in toc
    assert "  - [What's New?](#whats-new)" in toc


def test_assemble_toc_after_title() -> None:
    out = assemble(["# Doc\n\nText", "## Part\n\nMore"], toc_enabled=True)
    title_pos = out.index("# Doc")
    toc_pos = out.index("## Table of Contents")
    part_pos = out.index("## Part")
    assert title_pos < toc_pos < part_pos
    assert "[Doc](#doc)" not in out  # title excluded from its own TOC
    assert "[Part](#part)" in out
    assert assemble(["text"], toc_enabled=False) == "text\n"


def test_dedup_strips_running_header() -> None:
    pages = [
        "ACME Report\n# P1\nbody one\n7",
        "ACME Report\n# P2\nbody two\n8",
        "ACME Report\n# P3\nbody three\n9",
    ]
    result = dedup_furniture(pages, _same)
    assert all("ACME Report" not in page for page in result.pages)
    assert all(f"body {word}" in page for page, word in zip(result.pages, ("one", "two", "three")))
    assert any(entry["text"] == "ACME Report" for entry in result.removed)


def test_dedup_keeps_rare_lines() -> None:
    pages = ["# P1\nunique one", "# P2\nunique two"]
    result = dedup_furniture(pages, _same)
    assert result.removed == []
    assert result.pages == pages
