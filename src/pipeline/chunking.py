"""Hierarchical chunking — child chunks with heading paths + parent linkage.

Structure-aware RAG chunking for converted markdown (Phase 4):
- child chunks are paragraph/list/table-atomic units under a size cap,
  so a 12,000-token section no longer becomes one unusable embedding
- every child carries its heading path ("Chapter 7 > 7.4 Backpropagation")
  as metadata (optionally prepended to the embedded text — the cheap
  known retrieval win, mirroring LlamaIndex's MarkdownNodeParser)
- parent linkage (child → section text) enables small-to-big retrieval
  later without re-chunking

Atomicity rules: fenced code blocks and GFM tables are never split — a
chunk boundary inside a table destroys the structure the whole pipeline
exists to preserve. Blocks accumulate up to the token cap; a single
oversized atomic block becomes its own chunk (never truncated silently).
"""

import re
from dataclasses import dataclass, field

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = re.compile(r"^\s*```")
MAX_CHILD_TOKENS = 400  # child size cap (whitespace-split words)
MIN_CHILD_TOKENS = 40  # undersized fragments merge with neighbors when adjacent


@dataclass(frozen=True)
class ChildChunk:
    """One retrieval unit: atomic content + ancestry + parent linkage."""

    id: str
    text: str
    heading_path: tuple[str, ...] = ()
    parent_id: str | None = None
    page: int | None = None

    @property
    def heading_path_str(self) -> str:
        return " > ".join(self.heading_path)


@dataclass
class Section:
    """One heading section (the parent unit): id + heading + raw text."""

    id: str
    heading: str
    text: str
    children: list[ChildChunk] = field(default_factory=list)


def _token_len(text: str) -> int:
    return len(text.split())


def _split_blocks(section_text: str) -> list[str]:
    """Atomic blocks: fenced code and tables stay intact, else paragraphs.

    A blank line ends the current paragraph; fence/table lines accumulate
    into their own blocks regardless of internal blank lines.
    """
    lines = section_text.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    fence_buf: list[str] = []
    in_fence = False
    table_buf: list[str] = []
    in_table = False

    def _flush_paragraph() -> None:
        if current:
            blocks.append("\n".join(current).strip())
            current.clear()

    def _flush_table() -> None:
        nonlocal in_table
        if in_table and table_buf:
            blocks.append("\n".join(table_buf).strip())
            table_buf.clear()
            in_table = False

    for line in lines:
        if _FENCE.match(line):
            _flush_paragraph()
            _flush_table()
            fence_buf.append(line)
            in_fence = not in_fence
            continue
        if in_fence:
            fence_buf.append(line)
            continue
        if line.strip().startswith("|") and line.rstrip().endswith("|"):
            _flush_paragraph()
            in_table = True
            table_buf.append(line)
            continue
        if in_table:
            # table ends at a non-table, non-separator line
            if not line.strip():
                _flush_table()
                continue
            if re.match(r"^\s*\|?[\s:|-]+\|?[\s:|-]*$", line):
                table_buf.append(line)
                continue
            _flush_table()
        if not line.strip():
            _flush_paragraph()
            continue
        current.append(line)
    _flush_paragraph()
    _flush_table()
    if fence_buf:
        blocks.append("\n".join(fence_buf).strip())
    return [b for b in blocks if b]


def _sections_of(markdown: str) -> list[tuple[str, str, str]]:
    """(section_id, heading, section_text) in document order.

    Section text INCLUDES its heading line — parent retrieval returns
    what a reader would see. Section ids follow split_sections' ordering
    so old and new chunkers index compatibly.
    """
    sections: list[tuple[str, str, str]] = []
    heading = "(preamble)"
    buf: list[str] = []
    pending: list[tuple[str, str]] = []
    for line in markdown.splitlines():
        if m := _HEADING.match(line):
            if "".join(buf).strip():
                pending.append((heading, "\n".join(buf)))
            heading, buf = m.group(2).strip(), [line]
        else:
            buf.append(line)
    if "".join(buf).strip():
        pending.append((heading, "\n".join(buf)))
    for pos, (head, text) in enumerate(pending):
        sections.append((f"{pos:04d}", head, text))
    return sections


def chunk_children(
    markdown: str,
    *,
    docname: str = "doc",
    max_tokens: int = MAX_CHILD_TOKENS,
    min_tokens: int = MIN_CHILD_TOKENS,
) -> list[ChildChunk]:
    """Child chunks for one document (document order preserved).

    Oversized atomic blocks (one huge table/code block) become their own
    chunk — never split, never truncated. Undersized adjacent blocks in
    the same section merge up to the cap.
    """
    chunks: list[ChildChunk] = []
    for section_id, heading, section_text in _sections_of(markdown):
        blocks = _split_blocks(section_text)
        # Build heading ancestry from ALL headings in document order.
        merged: list[list[str]] = []
        for block in blocks:
            if (
                merged
                and _token_len(block) < min_tokens
                and _token_len("\n\n".join(merged[-1]) + "\n\n" + block) <= max_tokens
            ):
                merged[-1].append(block)
                continue
            if merged and _token_len("\n\n".join(merged[-1])) + _token_len(block) <= max_tokens:
                merged[-1].append(block)
                continue
            merged.append([block])
        # Oversized single blocks that ended up alone stay alone (intact).
        for pos, group in enumerate(merged):
            text = "\n\n".join(group).strip()
            if not text:
                continue
            chunks.append(
                ChildChunk(
                    id=f"{docname}:{section_id}:{pos:03d}",
                    text=text,
                    heading_path=_ancestry(markdown, heading),
                    parent_id=f"{docname}:{section_id}",
                )
            )
    return chunks


def _ancestry(markdown: str, heading: str) -> tuple[str, ...]:
    """Heading path from document root to `heading` inclusive.

    Walks headings in order maintaining a level stack; `heading` may be
    "(preamble)" which has no ancestors.
    """
    stack: list[tuple[int, str]] = []
    found: tuple[str, ...] | None = None
    for line in markdown.splitlines():
        if m := _HEADING.match(line):
            level, text = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            if text == heading and heading != "(preamble)":
                found = tuple(t for _, t in stack)
                break
    if found is not None:
        return found
    if heading == "(preamble)":
        return ()
    # heading not matched exactly (shouldn't happen): fall back to leaf name
    return (heading,)


def chunk_sections(markdown: str, *, docname: str = "doc") -> list[Section]:
    """Parent sections with their children attached (small-to-big basis)."""
    result: list[Section] = []
    children = chunk_children(markdown, docname=docname)
    for section_id, heading, section_text in _sections_of(markdown):
        section = Section(id=f"{docname}:{section_id}", heading=heading, text=section_text)
        section.children = [c for c in children if c.parent_id == section.id]
        result.append(section)
    return result


__all__ = [
    "MAX_CHILD_TOKENS",
    "MIN_CHILD_TOKENS",
    "ChildChunk",
    "Section",
    "chunk_children",
    "chunk_sections",
]
