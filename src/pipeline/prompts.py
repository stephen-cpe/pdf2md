"""Versioned prompt library — A.2/A.3/A.5 templates.

Pipeline stages import templates from here; prompt iteration edits ONLY this
module (never stage code). Each template carries a version identifier that
feeds compute_pipeline_version: same inputs → same version, any
prompt/model/config change → new job, never a silent resume mix.

Fabrication note: the verification template instructs the model to report
ADDED content under structure_issues — no schema change required. The diagram
templates ground every reinterpretation in the figure crop and require an
explicit non-convertible verdict rather than a guessed diagram.
"""

import hashlib
import json

TRANSCRIPTION_VERSION = "transcription-v2"
VERIFICATION_VERSION = "verification-v2"
DIAGRAM_VERSION = "diagram-v1"
DIAGRAM_VERIFICATION_VERSION = "diagram-verification-v1"

PROMPT_VERSIONS = {
    "transcription": TRANSCRIPTION_VERSION,
    "verification": VERIFICATION_VERSION,
    "diagram": DIAGRAM_VERSION,
    "diagram_verification": DIAGRAM_VERIFICATION_VERSION,
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

DIAGRAM_TEMPLATE = """You are a diagram-to-Mermaid reinterpreter. You are shown ONE cropped figure region from a PDF page.
Decide whether the figure can be faithfully re-expressed as Mermaid source, and if so produce it.
Rules:
- A figure is CONVERTIBLE when its structure maps to a Mermaid diagram type: flowcharts/process diagrams, sequence diagrams, class/ER diagrams, state machines, mind maps, timelines, Gantt charts, quadrants, and data charts (pie/xychart/sankey/radar).
- A figure is NOT convertible when it is a photograph, a raster illustration/artwork, a map, a logo, a chemical/molecular structure, a complex hand-drawn schematic, or any image whose meaning depends on pixels rather than relationships. Say so honestly — a guessed diagram is worse than an honest image.
- NEVER invent nodes, labels, numbers, or relationships that are not visible in the figure. Reinterpretation must be faithful, not plausible.
- The grounding text is OCR/native text from inside the figure region: trust it for exact labels and numbers; if it is empty, rely only on what you can read in the image.
- For data charts, put the extracted values in DATA so the caller can fall back to a grounded table if the Mermaid is rejected.
- Mermaid type must be one of: flowchart, sequenceDiagram, classDiagram, stateDiagram-v2, erDiagram, gantt, mindmap, timeline, journey, pie, gitGraph, quadrantChart, xychart-beta, sankey-beta, architecture-beta, radar-beta, kanban.
- Output ONLY the envelope. Markers at column 0, exact, case-sensitive.
Output format:
<<<DIAGRAM>>>
{"convertible": true | false, "type": "<mermaid type or empty>", "confidence": 0-100, "description": "<one-sentence summary of the figure>"}
<<<END_DIAGRAM>>>
<<<MERMAID>>>
[the complete Mermaid source, no ``` fences; empty when convertible is false]
<<<END_MERMAID>>>
<<<DATA>>>
{"columns": ["..."], "rows": [["...", "..."]]}
<<<END_DATA>>>
"""

DIAGRAM_VERIFY_TEMPLATE = """You are a strict Mermaid fidelity judge. Compare the candidate Mermaid source against the figure image and its grounding text.
Rules:
- Score COVERAGE 0-100: how faithfully the Mermaid captures the figure's structure, labels, connections, order, and values.
- FABRICATIONS are blocking: any node, edge, label, number, or relationship in the Mermaid with no basis in the image/grounding. A high-coverage candidate with invented content must NOT pass.
- MISSES: list visible structure the Mermaid drops or mangles (quote it).
- A Mermaid that renders but misrepresents the figure is a failure, not a pass.
- Output ONLY the verdict envelope: "pass" when coverage is high AND there are no fabrications AND no blocking misses, else "retry".
Output format (markers at line start, exact, case-sensitive):
<<<VERDICT>>>
{"coverage": 0-100, "misses": ["..."], "structure_issues": ["..."], "verdict": "pass" | "retry"}
<<<END_VERDICT>>>
"""


def compute_pipeline_version(
    *,
    prompts: dict[str, str] | None = None,
    agent_model: str,
    ocr_model: str,
    render_dpi: int,
    rolling_context_pages: int,
    coverage_threshold: int,
    coverage_floor: float,
    max_page_retries: int,
    thinking_transcribe: str,
    thinking_diagram: str,
    native_text_first: bool,
    native_text_min_words: int,
    toc_enabled: bool,
    fig_details: bool,
    diagram_to_mermaid: bool,
    diagram_min_confidence: int,
    diagram_verify: bool,
    diagram_fallback: str,
    diagram_keep_image: bool,
) -> str:
    """Deterministic job fingerprint: any input change → new version.

    Every behavior-changing JobOptions knob MUST be included; a config change
    that is absent here would let a resumed job silently mix pipeline
    behavior (the failure this fingerprint exists to prevent).
    """
    canonical = json.dumps(
        {
            "prompts": prompts if prompts is not None else PROMPT_VERSIONS,
            "agent_model": agent_model,
            "ocr_model": ocr_model,
            "render_dpi": render_dpi,
            "rolling_context_pages": rolling_context_pages,
            "coverage_threshold": coverage_threshold,
            "coverage_floor": coverage_floor,
            "max_page_retries": max_page_retries,
            "thinking_transcribe": thinking_transcribe,
            "thinking_diagram": thinking_diagram,
            "native_text_first": native_text_first,
            "native_text_min_words": native_text_min_words,
            "toc_enabled": toc_enabled,
            "fig_details": fig_details,
            "diagram_to_mermaid": diagram_to_mermaid,
            "diagram_min_confidence": diagram_min_confidence,
            "diagram_verify": diagram_verify,
            "diagram_fallback": diagram_fallback,
            "diagram_keep_image": diagram_keep_image,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "DIAGRAM_TEMPLATE",
    "DIAGRAM_VERIFICATION_VERSION",
    "DIAGRAM_VERIFY_TEMPLATE",
    "DIAGRAM_VERSION",
    "PROMPT_VERSIONS",
    "TRANSCRIPTION_TEMPLATE",
    "TRANSCRIPTION_VERSION",
    "VERIFICATION_TEMPLATE",
    "VERIFICATION_VERSION",
    "compute_pipeline_version",
]
