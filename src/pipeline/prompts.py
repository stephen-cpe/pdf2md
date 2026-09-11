"""Versioned prompt library — A.2/A.3/A.4 templates.

Pipeline stages import templates from here; prompt iteration edits ONLY this
module (never stage code). Each template carries a version identifier that
feeds compute_pipeline_version: same inputs → same version, any
prompt/model/config change → new job, never a silent resume mix.

Fabrication note: the verification template instructs the model to report
ADDED content under structure_issues — no schema change required.
"""

import hashlib
import json

TRANSCRIPTION_VERSION = "transcription-v2"
VERIFICATION_VERSION = "verification-v2"
QA_VERSION = "qa-v1"

PROMPT_VERSIONS = {
    "transcription": TRANSCRIPTION_VERSION,
    "verification": VERIFICATION_VERSION,
    "qa": QA_VERSION,
}

TRANSCRIPTION_TEMPLATE = """You are an expert document transcriber. Transcribe the page image to GitHub-Flavored Markdown.
Rules (verbatim with DEC-009 typo allowance):
- Reproduce ALL source text. Never summarize, paraphrase, or translate.
- You MAY correct obvious source spelling, grammar, or typos to standard form — log every correction in NOTES.
- GFM only: headings, paragraphs, lists, GFM pipe tables, fenced code blocks, block quotes, footnotes, LaTeX math in $...$ / $$...$$.
- Every figure/photo/chart/diagram region MUST become a placeholder token <!--FIG:page:INDEX:x0,y0,x1,y1--> on its own line, e.g. <!--FIG:page:1:100,200,300,400-->. Replace INDEX with the figure number (1, 2, 3, ...) — never emit the literal text "idx". Coordinate system (normative): x0,y0,x1,y1 are ABSOLUTE PIXELS relative to the rendered page PNG dimensions given below (origin top-left, x right, y down, within [0,width]/[0,height]). Never silently drop a figure.
- Continue multi-page constructs from context; exclude running headers/footers/page numbers from body output (list them separately).
- Preserve source reading order, including multi-column layouts (column-aware linearization).
- Uncertain characters: mark [?] and log in NOTES.
- The OCR reference below is ground truth for characters, NOT for structure.
Output format (normative textual envelope — markers at line start, exact, case-sensitive):
<<<MARKDOWN>>>
[the page's verbatim GFM]
<<<END_MARKDOWN>>>
<<<FIGURES>>>
[JSON array: {index, bbox [x0,y0,x1,y1 absolute pixels], alt, caption}]
<<<END_FIGURES>>>
<<<FURNITURE>>>
[JSON: header / footer / page-number strings, empty objects if none]
<<<END_FURNITURE>>>
<<<NOTES>>>
[continuations, uncertainties [?], typo corrections, decorative-element omissions]
<<<END_NOTES>>>
"""

VERIFICATION_TEMPLATE = """You are a strict verification judge. Compare the candidate Markdown against the page image and the OCR reference.
Rules:
- Score COVERAGE 0-100: how much of the page's content the Markdown captures (text, tables, formulas, figures, reading order).
- MISSES: list every content item present in the image/OCR but absent or mangled in the Markdown. Be specific (quote the missed text).
- TABLES: tabular data (ruled grids, aligned rows/columns, headed matrices) MUST appear as GFM pipe tables. Prose paragraphs where the image shows a ruled table are a blocking structure issue — quote the flattened region.
- FIGURES: every figure/photo/chart/diagram region visible in the image MUST have a <!--FIG:…--> placeholder in the Markdown. A FIGURES-worthy region with no placeholder is a blocking miss even when its caption text was transcribed as prose.
- FABRICATIONS go under STRUCTURE_ISSUES: any content in the Markdown with no basis in the image/OCR (hallucinated sentences, invented numbers, duplicated blocks passed off as distinct). A high-coverage page with fabrications must NOT pass.
- STRUCTURE_ISSUES also covers: wrong heading levels, broken tables, wrong reading order, missing figure placeholders, altered source wording beyond logged typo corrections.
- Output ONLY the verdict envelope: "pass" when coverage meets threshold AND no fabrications AND no blocking structure issues, else "retry".
Output format (markers at line start, exact, case-sensitive):
<<<VERDICT>>>
{"coverage": 0-100, "misses": ["..."], "structure_issues": ["..."], "verdict": "pass" | "retry"}
<<<END_VERDICT>>>
"""

QA_TEMPLATE = """You are a final quality editor for an assembled Markdown document transcribed from PDF pages.
Rules (DEC-003 binds this pass: Markdown SYNTAX may be fixed; transcribed source SEMANTICS and wording must never be altered):
- Check: heading hierarchy consistency, broken GFM constructs, leftover <!--FIG:...--> placeholders, orphaned fragments, duplicate sections from page joins.
- Emit corrections ONLY as section-anchored find/replace patches. A patch whose "replace" alters transcribed source wording (beyond whitespace/Markdown-syntax fixes) is forbidden — omit it.
- Every applied patch must be exactly reproducible from the summary log.
Output format (markers at line start, exact, case-sensitive):
<<<PATCHES>>>
[{"section": "<heading or location>", "find": "<exact source text>", "replace": "<corrected text>"}]
<<<END_PATCHES>>>
<<<SUMMARY>>>
[what was checked, patches applied, patches rejected with reasons]
<<<END_SUMMARY>>>
"""


def compute_pipeline_version(
    *,
    prompts: dict[str, str] | None = None,
    agent_model: str,
    ocr_model: str,
    embed_model: str,
    render_dpi: int,
    rolling_context_pages: int,
    coverage_threshold: int,
    coverage_floor: float,
    hybrid_routing: bool,
    max_page_retries: int,
    thinking_transcribe: str,
    thinking_qa: str,
    toc_enabled: bool,
    fig_details: bool,
) -> str:
    """Deterministic job fingerprint: any input change → new version."""
    canonical = json.dumps(
        {
            "prompts": prompts if prompts is not None else PROMPT_VERSIONS,
            "agent_model": agent_model,
            "ocr_model": ocr_model,
            "embed_model": embed_model,
            "render_dpi": render_dpi,
            "rolling_context_pages": rolling_context_pages,
            "coverage_threshold": coverage_threshold,
            "coverage_floor": coverage_floor,
            "hybrid_routing": hybrid_routing,
            "max_page_retries": max_page_retries,
            "thinking_transcribe": thinking_transcribe,
            "thinking_qa": thinking_qa,
            "toc_enabled": toc_enabled,
            "fig_details": fig_details,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "PROMPT_VERSIONS",
    "QA_TEMPLATE",
    "QA_VERSION",
    "TRANSCRIPTION_TEMPLATE",
    "TRANSCRIPTION_VERSION",
    "VERIFICATION_TEMPLATE",
    "VERIFICATION_VERSION",
    "compute_pipeline_version",
]
