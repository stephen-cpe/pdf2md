"""Synthesize a scanned (image-only) PDF for the eval corpus.

Renders a content-rich born-digital document to page images and wraps
those images in a new PDF with NO text layer — a faithful stand-in for
scanned paper: exactly what OCR+vision pipelines exist for and what
text-layer extractors cannot touch.

Usage: python eval/make_scanned_doc.py  (writes corpus/scanned_knowledge_handbook.pdf)
"""

import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "corpus" / "scanned_knowledge_handbook.pdf"
DPI = 200

CONTENT = [
    # page 1
    """The Knowledge Capture Handbook

A practical guide to preserving expert knowledge in engineering teams.

Chapter 1: Why Knowledge Capture Fails

When a senior engineer leaves a team, decades of undocumented
experience leave with them. Studies of organizational knowledge loss
estimate that replacing a departing expert costs between 200 and 300
percent of their annual salary once ramp-up, lost productivity, and
repeated mistakes are counted. The failure is rarely motivation:
engineers know they should document, but documentation always loses
to deadlines.

The most durable capture happens when writing is embedded in the work
itself rather than bolted on afterward. Checklists written during a
task survive; retrospective reports written three months later do not.

Key principles of durable capture:
- Write it while doing it, not after
- Capture decisions AND their rejected alternatives
- Prefer searchable notes over polished prose
- One fact per line, so future readers can cite precisely""",
    # page 2
    """Chapter 2: The Retention Model

We model knowledge retention with a decay function. If R is the
retained fraction after time t, and S is the strength of original
encoding, then:

R(t) = S * e^(-t / tau)

where tau (the retention constant) grows each time knowledge is
retrieved and used. Spaced retrieval practice raises tau; passive
re-reading does not. This is why a runbook that gets used weekly
outlives a wiki page nobody opens.

Recommended cadence per artifact class:

| Artifact | Review cadence | Owner |
|---|---|---|
| Runbook | After each incident | On-call lead |
| Architecture decision record | On relevant change | Tech lead |
| Onboarding checklist | Quarterly | Hiring manager |
| Postmortem | Never (append only) | Author |

The table shows the minimum viable cadence. Teams that skip reviews
watch their documents rot into misleading fiction, which is worse
than an empty page, because a wrong page actively damages trust.""",
    # page 3
    """Chapter 3: Practical Capture Patterns

Pattern 1: The Decision Ledger. Every non-trivial decision gets one
entry: context, options considered, choice, and the reason. The
rejected options matter most — six months later, someone will propose
the rejected option again, and the ledger saves the team from
re-litigating settled arguments.

Pattern 2: The Five-Minute Brain Dump. After solving any problem that
took longer than an hour, the solver spends five minutes writing what
they tried, what failed, and what worked. The dump is unpolished by
design; polish is the enemy of capture.

Pattern 3: The Walking Tour. New team members document the system as
they learn it. Their confusion is a sensor: every place a newcomer
gets lost is a place the documentation failed, and their fresh notes
fix what veterans no longer notice.

Conclusion

Knowledge capture is a system, not a virtue. Design the system so the
lazy path is also the correct path, and retention follows. Teams that
treat capture as part of the definition of done retain their experts'
knowledge even after the experts themselves have moved on.""",
]


def main() -> None:
    src = pymupdf.open()
    for page_text in CONTENT:
        page = src.new_page(width=612, height=792)
        rect = pymupdf.Rect(54, 54, 558, 738)
        page.insert_textbox(rect, page_text, fontsize=12.5, lineheight=1.35)

    scanned = pymupdf.open()
    for number in range(src.page_count):
        pix = src[number].get_pixmap(dpi=DPI)
        page = scanned.new_page(width=612, height=792)
        page.insert_image(page.rect, pixmap=pix)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    scanned.save(OUT)

    # verify: the scanned copy must have NO native text
    check = pymupdf.open(OUT)
    for number in range(check.page_count):
        words = len(check[number].get_text().split())
        assert words == 0, f"page {number + 1} leaked {words} words of native text"
    print(f"OK: {OUT} — {src.page_count} pages, zero native text")
    src.close()
    scanned.close()
    check.close()


if __name__ == "__main__":
    sys.exit(main())
