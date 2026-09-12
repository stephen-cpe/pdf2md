"""Assembly — concat, heading normalize, TOC, cross-page merges (6.1) + furniture dedup (6.2)."""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

_FIG_TOKEN = re.compile(r"<!--FIG:(?:page:\d+|\d+:\d+):[\d.,]+-->")

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|?[\s:|-]*$")
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+[.)])\s+\S")


def _is_table_row(line: str) -> bool:
    return _TABLE_ROW.match(line) is not None and _TABLE_SEP.match(line) is None


def normalize_headings(markdown: str) -> str:
    """Shift the whole hierarchy up when the doc never uses H1 (coherent levels)."""
    lines = markdown.splitlines()
    levels = [len(m.group(1)) for line in lines if (m := _HEADING.match(line))]
    if not levels or min(levels) <= 1:
        return markdown
    shift = min(levels) - 1
    return "\n".join(
        (f"{'#' * (len(m.group(1)) - shift)} {m.group(2)}" if (m := _HEADING.match(line)) else line)
        for line in lines
    )


def _github_slug(heading: str) -> str:
    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"\s+", "-", slug)


def build_toc(markdown: str) -> str:
    """GitHub-compatible TOC from headings (anchors match github.com slugs)."""
    entries = []
    for line in markdown.splitlines():
        if m := _HEADING.match(line):
            level, text = len(m.group(1)), m.group(2).strip()
            entries.append(f"{'  ' * (level - 1)}- [{text}](#{_github_slug(text)})")
    return "\n".join(entries)


def _ends_table_row(text: str) -> bool:
    nonempty = [line for line in text.splitlines() if line.strip()]
    return bool(nonempty) and _is_table_row(nonempty[-1])


def _starts_table_row(text: str) -> bool:
    nonempty = [line for line in text.splitlines() if line.strip()]
    if not nonempty:
        return False
    first = nonempty[0]
    return _is_table_row(first) or _TABLE_SEP.match(first) is not None


def _strip_leading_separator(text: str) -> str:
    lines = text.splitlines()
    idx = next((i for i, line in enumerate(lines) if line.strip()), None)
    if idx is not None and _TABLE_SEP.match(lines[idx]):
        return "\n".join(lines[:idx] + lines[idx + 1 :])
    return text


def _ends_list_item(text: str) -> bool:
    nonempty = [line for line in text.splitlines() if line.strip()]
    return bool(nonempty) and _LIST_ITEM.match(nonempty[-1]) is not None


def _starts_list_item(text: str) -> bool:
    nonempty = [line for line in text.splitlines() if line.strip()]
    return bool(nonempty) and _LIST_ITEM.match(nonempty[0]) is not None


def join_pages(pages: list[str]) -> str:
    """Concatenate with cross-page merges (6.1): split tables rejoin (dup
    separator dropped), continued lists stay tight (single newline)."""
    parts = [page.strip("\n") for page in pages if page.strip()]
    if not parts:
        return ""
    merged = [parts[0]]
    for nxt in parts[1:]:
        prev = merged[-1]
        if _ends_table_row(prev) and _starts_table_row(nxt):
            merged[-1] = prev + "\n" + _strip_leading_separator(nxt).lstrip("\n")
        elif _ends_list_item(prev) and _starts_list_item(nxt):
            merged[-1] = prev + "\n" + nxt.lstrip("\n")
        else:
            merged[-1] = prev + "\n\n" + nxt.lstrip("\n")
    return "\n".join(merged) + "\n"


def assemble(pages: list[str], toc_enabled: bool = True) -> str:
    """Full assembly (6.1): join → heading-normalize → optional TOC after H1.

    The title H1 itself is excluded from the TOC (it heads the page the TOC
    sits on — listing it is a self-link).
    """
    body = normalize_headings(join_pages(pages))
    if not toc_enabled:
        return body
    lines = body.splitlines(keepends=True)
    title_at = next((pos for pos, line in enumerate(lines) if line.startswith("# ")), None)
    if title_at is None:
        toc = build_toc(body)
        return ("## Table of Contents\n\n" + toc + "\n\n" + body) if toc else body
    head, tail = "".join(lines[: title_at + 1]), "".join(lines[title_at + 1 :])
    toc = build_toc(tail)
    if not toc:
        return body
    return head + "\n## Table of Contents\n\n" + toc + "\n" + tail


# --- 6.2 furniture dedup ---


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def candidate_lines(text: str) -> list[str]:
    """First 2 + last 2 non-empty lines — where running furniture lives."""
    nonempty = [line for line in text.splitlines() if line.strip()]
    return nonempty[:2] + nonempty[-2:]


@dataclass
class DedupResult:
    """Cleaned pages + furniture log (recorded in the report)."""

    pages: list[str]
    removed: list[dict[str, object]] = field(default_factory=list)


def dedup_furniture(
    page_texts: list[str],
    is_same: Callable[[str, str], bool],
    recurrence: float = 0.6,
) -> DedupResult:
    """Strip lines recurring on >=recurrence of pages (6.2, FR-QA-1/§6.2).

    is_same is the similarity predicate: exact-normalized by default, or a
    caller-supplied fuzzy matcher. Returns cleaned pages plus a log of
    {text, pages} for the conversion report. FIG placeholder lines are NEVER
    furniture (FR-AGT-3): unresolved figures must survive to fail QA loudly,
    and resolved image links differ per asset by construction.
    """
    total = len(page_texts)
    if total == 0:
        return DedupResult(pages=[])
    # Running furniture is a recurrence phenomenon: with fewer than two pages
    # every candidate line "recurs" at 100%, which would strip real content.
    if total < 2:
        return DedupResult(pages=list(page_texts))
    candidates: dict[str, str] = {}
    for text in page_texts:
        for line in candidate_lines(text):
            if _FIG_TOKEN.search(line):
                continue
            key = _normalize_line(line)
            candidates.setdefault(key, line)
    furniture: list[str] = []
    for key in candidates:
        hits = sum(
            1 for text in page_texts if any(is_same(key, other) for other in candidate_lines(text))
        )
        if hits / total >= recurrence:
            furniture.append(key)
    if not furniture:
        return DedupResult(pages=list(page_texts))
    cleaned = []
    for text in page_texts:
        kept = [
            line
            for line in text.splitlines()
            if not line.strip()
            or _FIG_TOKEN.search(line)
            or not any(is_same(_normalize_line(line), rep) for rep in furniture)
        ]
        cleaned.append("\n".join(kept).strip("\n") + "\n")
    removed = [
        {
            "text": rep,
            "pages": sum(
                1 for t in page_texts if rep in {_normalize_line(l) for l in t.splitlines()}
            ),
        }
        for rep in furniture
    ]
    return DedupResult(pages=cleaned, removed=removed)


def make_exact_similar(threshold: float = 0.9) -> Callable[[str, str], bool]:
    """Normalized-string similarity predicate (§6.2) without embeddings.

    Exact normalized matches always count; otherwise a difflib ratio at or
    above `threshold` (default 0.9) counts as the same running furniture.
    This keeps dedup fully local — no embedding model, no vector store.
    """
    import difflib

    def is_same(left: str, right: str) -> bool:
        left_n, right_n = _normalize_line(left), _normalize_line(right)
        if left_n == right_n:
            return True
        if not left_n or not right_n:
            return False
        return difflib.SequenceMatcher(None, left_n, right_n).ratio() >= threshold

    return is_same


__all__ = [
    "DedupResult",
    "assemble",
    "build_toc",
    "candidate_lines",
    "dedup_furniture",
    "join_pages",
    "make_exact_similar",
    "normalize_headings",
]
