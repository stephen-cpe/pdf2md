"""Hierarchical chunking — atoms, caps, ancestry, parent linkage."""

from src.pipeline.chunking import (
    MAX_CHILD_TOKENS,
    MIN_CHILD_TOKENS,
    chunk_children,
    chunk_sections,
)

DOC = """# Root Title

Intro paragraph one. It stands alone as the preamble.

## Section A

First paragraph of section A with enough words to be a proper block.

```python
def f():
    return 1
```

| col1 | col2 |
| --- | --- |
| a | b |
| c | d |

### Sub A1

Subsection text under A1.

## Section B

Second-level section text here.
"""


def test_children_have_heading_ancestry() -> None:
    kids = chunk_children(DOC, docname="doc")
    by_marker = {}
    for k in kids:
        if "First paragraph" in k.text:
            by_marker["a"] = k
        if "Subsection text" in k.text:
            by_marker["a1"] = k
        if "Second-level" in k.text:
            by_marker["b"] = k
    assert by_marker["a"].heading_path_str == "Root Title > Section A"
    assert by_marker["a1"].heading_path_str == "Root Title > Section A > Sub A1"
    assert by_marker["b"].heading_path_str == "Root Title > Section B"


def test_table_is_atomic() -> None:
    kids = chunk_children(DOC, docname="doc")
    tables = [k for k in kids if "| col1 | col2 |" in k.text]
    assert len(tables) == 1
    # the whole table lives in exactly one child, header to last row
    assert "| col1 | col2 |" in tables[0].text and "| c | d |" in tables[0].text


def test_code_fence_is_atomic_and_never_split() -> None:
    code = "# T\n\n```python\nline one\n\nline two\n```\n\ntail text\n" * 1
    kids = chunk_children(code, docname="d")
    fence = [k for k in kids if "```" in k.text]
    assert len(fence) == 1
    assert "line one" in fence[0].text and "line two" in fence[0].text


def test_large_section_splits_into_multiple_children() -> None:
    body = "\n\n".join(
        f"Paragraph {i} " + " ".join(f"word{i}{j}" for j in range(30)) for i in range(30)
    )
    md = f"# Big\n\n{body}\n"
    kids = chunk_children(md, docname="d")
    assert len(kids) > 1
    assert all(len(k.text.split()) <= MAX_CHILD_TOKENS for k in kids)


def test_undersized_blocks_merge() -> None:
    body = "\n\n".join(f"tiny{i} only few words" for i in range(6))
    md = f"# M\n\n{body}\n"
    kids = chunk_children(md, docname="d")
    assert len(kids) == 1  # all fragments under cap merge into one child
    assert "tiny0" in kids[0].text and "tiny5" in kids[0].text


def test_oversized_table_becomes_own_intact_chunk() -> None:
    rows = "\n".join(f"| r{i} | v{i} |" for i in range(300))
    md = "# T\n\n| h1 | h2 |\n| --- | --- |\n" + rows + "\n"
    kids = chunk_children(md, docname="d")
    tables = [k for k in kids if "| h1 | h2 |" in k.text]
    assert len(tables) == 1
    assert "| r0 | v0 |" in tables[0].text and "| r299 | v299 |" in tables[0].text
    # never split: no other child holds any fragment of the table
    assert all("| r" not in k.text or k is tables[0] for k in kids)


def test_parent_linkage_and_section_attach() -> None:
    sections = chunk_sections(DOC, docname="doc")
    ids = [s.heading for s in sections]
    assert ids == ["Root Title", "Section A", "Sub A1", "Section B"]
    total = sum(len(s.children) for s in sections)
    assert total == len(chunk_children(DOC, docname="doc"))
    # every child points at its parent section
    for s in sections:
        for c in s.children:
            assert c.parent_id == s.id


def test_preamble_children_have_no_ancestry() -> None:
    md = "Just a preamble paragraph without heading.\n\n# After\n\nbody\n"
    kids = chunk_children(md, docname="d")
    pre = [k for k in kids if "preamble paragraph" in k.text]
    assert pre and pre[0].heading_path == () and pre[0].parent_id == "d:0000"


def test_chunk_order_preserved() -> None:
    kids = chunk_children(DOC, docname="doc")
    positions = []
    for k in kids:
        for needle in ("Intro paragraph", "First paragraph", "Subsection text", "Second-level"):
            if needle in k.text:
                positions.append(needle)
    assert positions == ["Intro paragraph", "First paragraph", "Subsection text", "Second-level"]


def test_min_tokens_constant_is_sane() -> None:
    assert 0 < MIN_CHILD_TOKENS < MAX_CHILD_TOKENS
