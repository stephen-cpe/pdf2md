# Software Requirements Specification (SRS)

## PDF2MD-Agent — Agentic PDF → GFM Converter with Diagram→Mermaid Reinterpretation

| Field | Value |
|---|---|
| **Status** | Draft. The source code is the truth; this document describes what it does. |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Overall Description](#2-overall-description)
3. [System Architecture](#3-system-architecture)
4. [Functional Requirements](#4-functional-requirements)
5. [External Interface Requirements](#5-external-interface-requirements)
6. [Data Requirements](#6-data-requirements)
7. [Non-Functional Requirements](#7-non-functional-requirements)
8. [Error Handling & Recovery](#8-error-handling--recovery)
9. [Constraints & Assumptions](#9-constraints--assumptions)
10. [Acceptance Criteria](#10-acceptance-criteria)
11. [Out of Scope](#11-out-of-scope)
12. [Glossary](#12-glossary)
13. [Appendix A — Prompt Contracts](#appendix-a--prompt-contracts)
14. [Appendix B — Configuration Reference](#appendix-b--configuration-reference)

---

## 1. Introduction

### 1.1 Purpose

This document specifies **PDF2MD-Agent**, a locally run application that converts a single PDF file into a **GitHub-Flavored Markdown (GFM)** document that reproduces the source as faithfully as possible — including headings, body text, tables, formulas, code blocks, lists, and footnotes — and, as its **primary capability**, reinterprets the document's **figures — diagrams, flowcharts, charts, graphs, schematics — as Mermaid source** wherever that can be done faithfully. Figures that cannot (or should not) be reinterpreted fall back to an extracted image with AI-generated alt-text, or to a grounded data table for data charts.

The conversion is **agentic**: a large multimodal LLM (GLM-5.3-Flash, 1M-token context, vision + tool use) reads the rendered page image, cross-checks against the output of a dedicated OCR model (GLM-OCR), and iteratively writes and self-corrects the Markdown until the page passes both the agent verdict and a deterministic token-recall floor. A second, dedicated per-figure stage then crops each flagged figure region and reinterprets it as Mermaid, gated by deterministic validation and a vision verification pass. This is explicitly a replacement for prior Tesseract-based pipelines that dropped images and mangled layout.

Two objective quality gates back every verified page: (1) the agent-computed coverage score (cross-checked against the page image), and (2) a **deterministic coverage floor** — the fraction of OCR/native-text tokens present in the produced Markdown must meet a configurable threshold (default 80%), making omissions measurable and fabrication incapable of raising the score. Figure reinterpretations carry their own gates: a Mermaid type allowlist plus structural validation, and (by default) a vision verifier that compares the Mermaid source against the figure crop and fails fabrications.

**Guiding principle:** *Completeness and fidelity over speed. Never invent; reinterpret only what the figure actually shows.* The system may take as long as needed; it must never silently skip content, and a guessed diagram is worse than an honest image.

### 1.2 Scope

**In scope:**

- One PDF at a time, uploaded through a local web interface.
- Full-document conversion to a single top-level `.md` file plus a per-document folder holding the extracted-image `assets/` and the conversion report.
- **Diagram→Mermaid reinterpretation** of every flagged figure region: crop → grounding → conversion → validation → vision verification → tiered output (Mermaid / data table / image).
- The full agentic transcription path: GLM-5.3-Flash (Ollama Cloud) with GLM-OCR (local Ollama) as the OCR ground-truth tool.
- Two quality gates per page: agent verification verdict AND a deterministic token-recall coverage floor.
- User-specified output directory on the local machine; many documents may share one output directory without overwriting each other.
- Job persistence, checkpointing, and resume in PostgreSQL.
- Live progress monitoring in the web UI (page N of M, current stage, live Markdown preview).

**Out of scope** — see §11:

- Batch/multi-PDF processing, multi-user auth, cloud deployment, output formats other than GFM (HTML export is a stretch goal), editing UI, OCR model fine-tuning.
- Rendering Mermaid to images for visual diffing (no local Node/mermaid-cli dependency): verification is vision-model based.

### 1.3 Definitions & References

See §12 (Glossary). Model references:

- GLM-5.3-Flash — 320B-A18B natively multimodal MoE, 1M-token context, vision + tools + thinking; used via **Ollama Cloud** (`glm-5.3-flash:cloud`). <https://ollama.com/library/glm-5.3-flash>, <https://huggingface.co/zai-org/GLM-5.3-Flash>
- GLM-OCR — compact GLM-V-based OCR model (CogViT encoder + GLM-0.5B decoder), SOTA on OmniDocBench V1.5 (94.62), strong on tables, formulas, and complex layouts; used via **local Ollama** (`glm-ocr`). No cloud variant exists; local execution is a hard requirement. <https://ollama.com/library/glm-ocr>, <https://huggingface.co/zai-org/GLM-OCR>
- GitHub-Flavored Markdown Spec: <https://github.github.com/gfm/>

---

## 2. Overall Description

### 2.1 Product Perspective

PDF2MD-Agent is a standalone, self-hosted application composed of:

- A **FastAPI** backend (Python 3.14) exposing a REST + WebSocket API.
- A **minimal web UI** (served by FastAPI) for upload, configuration, monitoring, and result retrieval.
- A **conversion pipeline worker** that renders PDF pages, calls the OCR and agent models, reinterprets figures as Mermaid, and assembles the Markdown output.
- **PostgreSQL** for job/page state and audit history.
- **Ollama** (local daemon) hosting `glm-ocr`; **Ollama Cloud** hosting `glm-5.3-flash`, both accessed through the Ollama API. These are two **separate endpoints** (Appendix B): the local daemon (`OLLAMA_LOCAL_URL`, no auth) and the Cloud API (`OLLAMA_CLOUD_URL`, `Authorization: Bearer <OLLAMA_API_KEY>`). The application never routes a Cloud-model call through the local daemon — routing is explicit, deterministic, and testable, independent of any interactive `ollama signin` state.

### 2.2 User Class

A single technical user (developer/power user) running the app on their own desktop/laptop on Windows 11 native. No authentication; the server binds to localhost by default.

### 2.3 Operating Environment

| Component | Requirement |
|---|---|
| OS | Windows 11 native |
| Python | 3.14.x |
| Ollama | Latest stable, running locally with `glm-ocr` pulled |
| Ollama Cloud | Active Pro subscription + API key for `glm-5.3-flash:cloud` |
| PostgreSQL | 18.x native Windows service (local install on localhost:5432; 18.6 verified — 16+ compatible) |
| Disk | ≥ 2× size of PDF free for rendered page images + assets |
| Network | Internet access required for Ollama Cloud calls |

### 2.4 Key Design Notes

| Note | Content |
|---|---|
| **Per-page reference routing** | A page with a substantial native text layer uses it as the character reference and skips OCR (exact characters, zero model calls, no degenerate-loop exposure); sparse/scanned pages call local `glm-ocr`. Retries keep the native reference and only re-render at higher DPI. Config-gated (`NATIVE_TEXT_FIRST`, default on) and part of the `pipeline_version` fingerprint. |
| **Figures are reinterpreted as Mermaid (primary capability)** | Every flagged figure region is cropped, optionally grounded in the text inside it, converted to Mermaid by a dedicated Cloud call, validated against a type allowlist, and (by default) vision-verified against the crop. Fabricated nodes/edges/values fail verification. Non-convertible figures fall back to image+alt-text or a grounded data table — never silently dropped. |
| **Tiered fallback, never silent loss** | Mermaid when validated+verified; OCR-grounded GFM table + image for data charts that fail conversion; image + alt-text + caption for photos/illustrations/maps or crop failures. The conversion report records per-figure status and the overall conversion rate. |
| Agent reads **rendered page images**, not extracted text | Vision preserves layout, figures, and reading order — and is the only path that works for scanned pages. OCR text is a character reference, never the writer. |
| **Two quality gates: judge + floor** | The agent verdict alone is LLM-grading-LLM. Every verified page also passes a deterministic coverage floor: the fraction of OCR-reference tokens present in the produced Markdown must be ≥ `COVERAGE_FLOOR_TOKENS` (default 80%). Omission becomes measurable; fabrication cannot raise the score. Short OCR references (< `COVERAGE_FLOOR_MIN_OCR_TOKENS`) are unmeasurable and never gate. |
| GLM-OCR as a **tool**, not the writer | OCR gives high-accuracy character-level ground truth for text/tables/formulas; the agent decides structure, ordering, and prose assembly. Best of both. |
| Page-by-page with rolling context | Even with 1M context, page-wise processing gives checkpointing, resumability, and bounded failure. The agent still receives prior-page context for continuity (heading levels, running lists, multi-page tables). |
| Images extracted **and** AI-captioned | GitHub-compatible relative links + meaningful alt-text for accessibility and search. |
| PostgreSQL checkpoints every page | Long jobs must survive crashes; a crash at page 412 of 500 must not lose work. |
| Direct Cloud API, never CLI login state | A background worker cannot rely on interactive `ollama signin` state. Auth is exclusively `OLLAMA_API_KEY` against `OLLAMA_CLOUD_URL`. |
| Explicit dual-client routing: local vs Cloud | `glm-ocr` → local daemon, no auth; `glm-5.3-flash` → Cloud client with Bearer key. Prevents endpoint mix-ups and key leakage. |
| Text and visual/layout fidelity equally important | Markdown normalization may fix syntax but must never alter transcribed source semantics or wording. |
| Verified-page immutability | A page checkpointed `verified`/`needs_review` is never implicitly re-run. Re-transcription requires explicit invalidation or a new job. |
| Pipeline versioning + content hashes; config change ⇒ new job | Each job records a `pipeline_version` hash (prompt versions + models + DPI + coverage threshold + floor + diagram config); per-page hashes pin provenance. |
| Figure bbox = absolute pixels of the rendered PNG | Origin top-left, x right, y down. The prompt includes exact image dimensions; the cropper uses the same render — no unit conversion. |
| Per-page model telemetry is first-class | DPI, image dims/bytes, model, effort, tokens, latency, retries, verification score recorded per page for cost/accuracy tuning. |
| Rolling context stays config-driven | `ROLLING_CONTEXT_PAGES` is configurable, never hard-coded. |
| Agent may correct obvious source typos, logged | The model's spelling-correction prior beats prompt wording; every correction is logged in NOTES. Genuine ambiguity is still marked `[?]`. |
| Per-document output folders | Deterministic asset names collide across documents, so each document gets `<output>/<doc>/assets/` + report while the `.md` stays top-level at `<output>/<doc>.md` for navigation. |
| Explicit `INDEX` figure tokens + malformed-token hard-fail | The prompt shows `<!--FIG:page:INDEX:…-->` with an example and forbids the literal text `idx`. Any remaining `<!--FIG:` marker (well-formed or malformed) fails integrity loudly. |
| **Mermaid is GitHub-native** | Mermaid fenced blocks render directly in GFM on GitHub, so a reinterpreted figure is searchable and diffable text rather than an opaque raster. |

---

## 3. System Architecture

### 3.1 Component Diagram (logical)

```
┌────────────────────────────────────────────────────────────────────┐
│                          Web UI (browser)                          │
│    upload · configure · progress (WebSocket) · preview · download  │
└───────────────▲────────────────────────────────────────────────────┘
                │ REST + WebSocket (localhost)
┌───────────────┴────────────────────────────────────────────────────┐
│                        FastAPI Application                         │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────────────┐  │
│  │  Job API     │  │ Progress Hub  │  │  Output Writer          │  │
│  │  (CRUD)      │  │ (WS events)   │  │  (md + assets to path)  │  │
│  └──────┬───────┘  └──────▲────────┘  └───────────▲─────────────┘  │
│         │                 │                       │                │
│  ┌──────▼─────────────────┴───────────────────────┴─────────────┐  │
│  │              Conversion Pipeline Worker (async task)         │  │
│  │  1 Preflight → 2 Render → 3 OCR → 4 Agent transcribe →      │  │
│  │  5 Verify + coverage floor → 6 Diagram→Mermaid per figure →  │  │
│  │  7 Image extract+caption → 8 Assemble → 9 Integrity + report │  │
│  └──────┬──────────────┬──────────────┬───────────┬─────────────┘  │
└─────────┼──────────────┼──────────────┼───────────┼────────────────┘
          │              │              │           │
   ┌──────▼─────┐ ┌──────▼─────┐ ┌─────▼─────┐ ┌───▼──────────────┐
   │ PostgreSQL │ │  Ollama    │ │  Ollama   │ │  Local FS        │
   │ jobs/pages │ │  (local)   │ │  Cloud    │ │ workspace +      │
   │ images/    │ │  glm-ocr   │ │ glm-5.3-  │ │ output .md +     │
   │ events     │ │            │ │ flash     │ │ assets/          │
   └────────────┘ └────────────┘ └───────────┘ └──────────────────┘
```

### 3.2 Conversion Pipeline (per job)

| Stage | Name | Description |
|---|---|---|
| 1 | **Preflight** | Validate PDF (readable, not encrypted, page count, size limits). Detect born-digital vs scanned. Record metadata. |
| 2 | **Render** | Render every page to PNG at configurable DPI (default 200, auto-bump toward 400 for dense/small-font pages, with low-DPI fallback for degenerate OCR loops). Store in job workspace. |
| 3 | **OCR pass** | Reference routing: a page with a substantial native text layer uses that layer as the character reference and skips OCR entirely (exact characters, zero model calls). Pages with sparse or absent native text (scans, figure-only pages) call local `glm-ocr` with its document-parsing prompts (`Text Recognition:`, `Table Recognition:`, `Formula Recognition:`). The per-page reference kind is recorded in the page's omissions. |
| 4 | **Agent transcription** | For each page, GLM-5.3-Flash receives: (a) the page image, (b) the GLM-OCR output, (c) rolling context (last N pages of produced Markdown + document outline so far). It produces the page's GFM, resolves reading order, merges multi-page constructs, and flags every figure region. |
| 5 | **Page verify + coverage floor** | The agent compares the Markdown against the page image + OCR text (self-check pass), emitting a coverage verdict; **independently**, the deterministic floor computes token recall of the reference text in the produced Markdown. A page verifies only when BOTH the verdict passes at threshold AND the floor holds. Failures retry with higher DPI and/or explicit "you missed X" feedback (max R retries, default 2), then flag `needs_review`. |
| 6 | **Diagram → Mermaid** | For every figure region flagged on a verified page: crop the region; ground it in native text inside the bbox (best effort); convert with a dedicated Cloud call; validate the Mermaid type against the allowlist and its structure; vision-verify the Mermaid against the crop. The result is a `DiagramResult` carrying Mermaid source, type, confidence, description, and (for data charts) a grounded table payload. |
| 7 | **Image handling** | Reuse embedded-image bytes only when the object's placed geometry matches the agent-flagged region (overlap + comparable size); otherwise crop the region from the rendered page. Save per-document to `<doc>/assets/`. Emit the tiered representation: Mermaid (with collapsible original image when configured), grounded data table + image, or image + alt-text/caption. |
| 8 | **Assemble** | Concatenate pages; deduplicate running headers/footers/page numbers (local normalized + fuzzy similarity); normalize heading hierarchy; fix cross-page tables/lists; build optional TOC; rewrite image links to the per-document relative path. |
| 9 | **Integrity + report** | GFM lint (via `mdformat`/`pymarkdownlnt`) and link/asset integrity check (hard-fail on unresolved `FIG` token or missing asset). Write the conversion report including the per-figure diagram conversion summary and overall conversion rate. |

### 3.3 Technology Stack (normative)

| Layer | Choice |
|---|---|
| Language | Python 3.14 |
| Web framework | FastAPI + Uvicorn |
| Front-end | Server-served static SPA (vanilla JS + minimal CSS; no build step required) |
| PDF rendering/extraction | PyMuPDF (`pymupdf`) — page rasterization, native image extraction, metadata, text-layer detection |
| LLM access | `ollama` Python client — local daemon for `glm-ocr`; Ollama Cloud (API key) for `glm-5.3-flash:cloud` |
| ORM / migrations | SQLAlchemy 2.x (async) + Alembic |
| DB | PostgreSQL 18.x (`asyncpg`, 18.6 verified on Windows 11) |
| Diagram output | Mermaid fenced code blocks (GitHub-native rendering); validated against a configurable type allowlist |
| Markdown QA | `pymarkdownlnt` (GFM lint), `mdformat-gfm` (normalization) |
| Furniture similarity | `difflib` normalized fuzzy matching (no vector store dependency) |
| Task execution | `asyncio` background task per job (single worker, one job at a time) |
| Config | `.env` + `pydantic-settings`; sample file `.env.example` |
| Packaging | `pyproject.toml` (pip) — Windows 11 native installs only |

---

## 4. Functional Requirements

Requirement IDs use `FR-<area>-<n>`. Priority: **M**ust / **S**hould / **C**ould (MoSCoW).

### 4.1 Job Management (JOB)

| ID | Priority | Requirement |
|---|---|---|
| FR-JOB-1 | M | The system SHALL accept exactly one PDF upload per job via the web UI (multipart) with a configurable max size (default 500 MB). |
| FR-JOB-2 | M | The system SHALL let the user specify a local output directory path for the converted files before starting a job, and SHALL validate that the path exists and is writable (creating it on request). |
| FR-JOB-3 | M | The system SHALL process at most one job at a time; new uploads while a job is running SHALL be rejected with a clear message (409). |
| FR-JOB-4 | M | The system SHALL persist job state (status, current stage, current page, per-page results, config snapshot) to PostgreSQL after **every page**, so that a crash or restart loses at most one page of work. |
| FR-JOB-5 | M | The system SHALL support resuming an interrupted job from its last checkpoint via UI action (restart endpoint, resumable from the last non-terminal page). |
| FR-JOB-6 | M | The system SHALL support user-initiated cancel; a cancelled job's partial output SHALL remain resumable. Pause/resume mid-job SHALL also be supported. |
| FR-JOB-7 | S | The system SHALL keep a job history (last 50 jobs) with links to outputs and conversion reports. |
| FR-JOB-8 | M | Job statuses: `queued`, `preflight`, `rendering`, `ocr`, `transcribing`, `assembling`, `completed`, `failed`, `cancelled`, `paused`. Illegal transitions are rejected, never silently applied. (`qa` is retained as a legacy status for pre-existing job rows; the current pipeline does not enter it.) |

### 4.2 Preflight & Rendering (PDF)

| ID | Priority | Requirement |
|---|---|---|
| FR-PDF-1 | M | The system SHALL validate the uploaded file is a parseable PDF; corrupt or password-protected files SHALL fail preflight with an explanatory error. |
| FR-PDF-2 | M | The system SHALL record page count, dimensions, PDF metadata (title, author), and whether a text layer exists (born-digital vs scanned heuristic). |
| FR-PDF-3 | M | The system SHALL render each page to PNG at a configurable DPI (default 200). |
| FR-PDF-4 | S | The system SHALL automatically re-render individual pages at higher DPI (up to 400) when verification reports low legibility, and fall back to lower DPIs plus text-layer extraction when OCR degenerates into loops. |
| FR-PDF-5 | M | Rendered page images SHALL be stored in a per-job workspace directory and cleaned up per retention policy (default: delete on success, keep on failure; Windows file-lock contention is retried with backoff, then deferred — never failing a completed job). |
| FR-PDF-6 | M | The system SHALL expose per-page native-text access (whole page) and bbox-clipped native-text access for figure grounding (text intersecting the figure region only, converting render-pixel bboxes to PDF points), without crashing on corrupt or undecodable pages (undecidable ⇒ return empty text; the figure still falls back to image). |

### 4.3 OCR Pass (OCR)

| ID | Priority | Requirement |
|---|---|---|
| FR-OCR-1 | M | The system SHALL obtain a per-page character reference for every page: the native text layer when it is substantial (≥ `NATIVE_TEXT_MIN_WORDS` words, with a character fallback for space-less scripts) and `NATIVE_TEXT_FIRST` is enabled (default on), otherwise the **local** Ollama `glm-ocr` model using its document-parsing prompt contract (Appendix A.1). |
| FR-OCR-2 | M | The character reference (native text or OCR) SHALL be persisted with the page record and used by the prompt contract and the coverage floor, without changing the page's verification semantics. |
| FR-OCR-3 | M | If local Ollama or `glm-ocr` is unreachable when a page actually needs OCR, preflight SHALL fail with actionable instructions (e.g. `ollama pull glm-ocr`). |
| FR-OCR-4 | S | The system SHOULD run OCR as a look-ahead pipeline (OCR page N+1 while the agent transcribes page N) to reduce wall-clock time without changing output. |
| FR-OCR-5 | M | On retry, a page routed to the native reference SHALL keep that exact reference and only re-render at the higher DPI; only OCR-routed pages SHALL re-OCR. The reference kind (`native`/`ocr`/`vision`) SHALL be recorded per page and surfaced in the report. |
| FR-OCR-6 | M | `OCR_ENABLED=false` SHALL disable the local OCR stage entirely: pages without a viable native layer transcribe vision-only with an empty reference (never gated by the floor), and no `glm-ocr` call SHALL be made. The flag SHALL participate in the `pipeline_version` fingerprint. |

### 4.3a Deterministic Coverage Floor (FLR)

| ID | Priority | Requirement |
|---|---|---|
| FR-FLR-1 | M | The system SHALL compute, for every page attempt, a deterministic coverage score: the fraction of reference tokens (native text when the page routed native, else OCR output) present in the produced Markdown, after deterministic Markdown-syntax-noise stripping (fences, heading markers, link/image targets with alt/label text preserved). |
| FR-FLR-2 | M | A page SHALL be considered verified only when the agent verdict passes at threshold AND `floor_score ≥ COVERAGE_FLOOR_TOKENS` (default 80%); a floor failure triggers the same retry cycle as a verdict failure, with a corrective instruction naming the deficit. |
| FR-FLR-3 | M | OCR references with fewer than `COVERAGE_FLOOR_MIN_OCR_TOKENS` tokens (default 30) are unmeasurable — the floor returns no score and never gates the page (the verdict alone stands). `COVERAGE_FLOOR_TOKENS=0` disables the floor. |
| FR-FLR-4 | M | The floor score SHALL be recorded per page in `omissions.floor_score` (verified and needs_review checkpoints) and surfaced in the conversion report. Fabrication cannot raise the floor score — only reference tokens found in the Markdown count. |

### 4.4 Agent Transcription (AGT)

| ID | Priority | Requirement |
|---|---|---|
| FR-AGT-1 | M | For each page, the system SHALL send `glm-5.3-flash` (Ollama Cloud, via its dedicated client — see §5.3) a multimodal request containing: the page image, the page's GLM-OCR output, the rolling context (document outline + last N pages of produced Markdown, N default 3, configurable — must not be hard-coded), and the transcription prompt contract (Appendix A.2). |
| FR-AGT-2 | M | The agent SHALL output GitHub-Flavored Markdown reproducing all page content: headings (correct hierarchy), paragraphs, ordered/unordered lists, GFM pipe tables, code blocks with language fences, block quotes, footnotes, and LaTeX math in `$...$` / `$$...$$` (GitHub-renderable). |
| FR-AGT-3 | M | The agent SHALL mark every figure/photo/chart/diagram region with a structured placeholder token (`<!--FIG:page:INDEX:bbox-->` where `INDEX` is 1, 2, 3, … — never the literal text `idx`) — figures MUST NOT be silently dropped. Decorative elements (rules, backgrounds) are the only permitted omissions and must be listed in the page log. **Coordinate system (normative):** `x0,y0,x1,y1` are **absolute pixels relative to the rendered page PNG's dimensions** (origin = top-left, x right, y down; values within `[0, width]`/`[0, height]`). The transcription prompt MUST include the exact rendered image dimensions so the agent can anchor its estimates; the crop step (FR-IMG-1) uses the same render, so no unit conversion occurs. |
| FR-AGT-3a | M | The verification call (Appendix A.3) SHALL use the same textual-envelope output contract as the transcription call (Appendix A.2) — it is a Cloud-model call and therefore cannot rely on server-enforced JSON schema output. |
| FR-AGT-4 | M | The agent SHALL correctly continue multi-page constructs: tables split across pages are merged into one table; lists, paragraphs, and section numbering continue seamlessly; running headers/footers/page numbers are excluded from body output (recorded separately). |
| FR-AGT-5 | M | The agent SHALL preserve source reading order, including multi-column layouts (column-aware linearization, via prompt rules). |
| FR-AGT-6 | M | The agent SHALL NOT summarize, paraphrase, translate, or "improve" source text, except that it MAY correct obvious source spelling/grammar/typos to standard form — every correction is logged in NOTES. Verbatim transcription otherwise; uncertain characters marked `[?]` and logged. |
| FR-AGT-7 | M | **Verification:** after producing a page, the agent SHALL perform a self-check pass (page image + OCR vs its Markdown) and emit a verdict in the Appendix A.3 envelope format with a coverage score 0–100 and a list of misses. A page verifies only when the verdict passes (score ≥ threshold, default 95) AND the deterministic coverage floor holds (FR-FLR-2). Either failure triggers a retry with the miss list injected; max retries default 2, then the page is flagged `needs_review` in the report (never silently accepted). |
| FR-AGT-8 | M | Every model call SHALL implement timeout, exponential-backoff retry (default 3 attempts) for transient network/5xx errors, and rate-limit (429) handling with respect for `Retry-After`. Sustained outage pauses the job (resumable), never fails it. |
| FR-AGT-9 | S | Token usage (prompt/completion) per call SHALL be recorded per page for cost visibility in the report. |
| FR-AGT-10 | S | Per-page model telemetry SHALL additionally record: `render_dpi`, `image_width`, `image_height`, `image_bytes`, `model`, `thinking_effort`, `latency_ms`, `retry_count`, and `verification_score` alongside token usage, enabling empirical DPI/cost/accuracy tuning. |

### 4.5 Image Handling (IMG)

| ID | Priority | Requirement |
|---|---|---|
| FR-IMG-1 | M | The system SHALL extract every figure referenced by a `FIG` placeholder: from the native PDF object (original resolution) only when its placed geometry matches the agent-provided bbox (overlap plus comparable size — a small icon inside a large flagged region must not stand in for it), otherwise cropped from the rendered page using the agent-provided bbox (absolute pixels in the rendered PNG's coordinate space per FR-AGT-3). |
| FR-IMG-2 | M | Images SHALL be saved per document to `<output>/<docname>/assets/` as PNG (or original format if JPEG) with deterministic names: `page-{page:03d}-img-{n:02d}.png`. The per-document folder is required because these names collide across documents sharing one output directory. Native-sourced rasters larger than 1600px on the long edge SHALL be downscaled to 1600px (format preserved; at-or-under images stay byte-identical). |
| FR-IMG-3 | M | GLM-5.3-Flash SHALL generate for each image: (a) concise alt-text (≤ 125 chars) and (b) where the source shows a caption, that caption verbatim; output as `![alt](<docname>/assets/…)` followed by any caption in *italics* (matching the source's caption text). |
| FR-IMG-4 | S | For charts/diagrams, the agent SHOULD additionally produce a longer description as an HTML `<details>` block beneath the image (configurable, default on) so information in the graphic survives text-only contexts. |
| FR-IMG-5 | M | All image links in the final Markdown SHALL be relative (`<docname>/assets/...`) so the output folder stays portable/GitHub-compatible. |
| FR-IMG-6 | M | A final integrity check SHALL verify every image link resolves to an existing file, every extracted file is referenced, and no `<!--FIG:` marker of any shape remains (well-formed or malformed, e.g. a literal `idx`); discrepancies fail the job (FR-QA-3). |
| FR-IMG-7 | M | A figure whose Mermaid reinterpretation is accepted SHALL emit a ` ```mermaid ` fenced block at the placeholder site; when `DIAGRAM_KEEP_IMAGE=true`, the original image SHALL be retained in a collapsible `<details>` block beneath it. |

### 4.6 Diagram → Mermaid (DGM)

| ID | Priority | Requirement |
|---|---|---|
| FR-DGM-1 | M | The system SHALL run a dedicated per-figure conversion stage after page verification: crop each flagged figure region, and (best effort) ground the conversion in native text clipped to the figure's bbox only (never the whole page). |
| FR-DGM-2 | M | The conversion call (Appendix A.5) SHALL be a Cloud call using the same textual-envelope contract as the other Cloud calls, returning a convertible verdict, a Mermaid type, a confidence score, a description, and (for data charts) a grounded data payload. |
| FR-DGM-3 | M | A candidate Mermaid SHALL pass deterministic validation before use: a recognized type on the first meaningful line, membership in the configurable allowlist (`DIAGRAM_ALLOWED_TYPES`), a non-empty body, and no leftover pipeline markers. |
| FR-DGM-4 | M | When `DIAGRAM_VERIFY=true` (default), a second vision call SHALL compare the candidate Mermaid against the figure crop; a verdict below threshold, or any fabricated node/edge/label/value, SHALL reject the candidate. |
| FR-DGM-5 | M | The system SHALL apply a tiered fallback, never silently dropping a figure: (a) validated+verified Mermaid; (b) for data charts, an OCR/agent-grounded GFM data table plus the image; (c) image + alt-text + caption. |
| FR-DGM-6 | M | The chosen representation, Mermaid source, diagram type, and confidence SHALL be persisted on the figure's `images` row; the conversion report SHALL list per-figure status and the overall conversion rate. |
| FR-DGM-7 | M | Conversion SHALL be config-gated (`DIAGRAM_TO_MERMAID`, default **on**) and every diagram behavior knob (`DIAGRAM_TO_MERMAID`, `DIAGRAM_MIN_CONFIDENCE`, `DIAGRAM_VERIFY`, `DIAGRAM_FALLBACK`, `DIAGRAM_KEEP_IMAGE`) SHALL participate in the `pipeline_version` fingerprint — a conversion-config change is a new job, never a silent resume mix. |

### 4.7 Assembly, Integrity & Report (QA)

| ID | Priority | Requirement |
|---|---|---|
| FR-QA-1 | M | The system SHALL assemble per-page Markdown into one document: strip deduped running headers/footers, merge cross-page constructs, normalize heading levels to a single coherent hierarchy, and optionally (config, default on) insert a generated TOC after the title. |
| FR-QA-3 | M | The system SHALL lint the final document as GFM (`pymarkdownlnt`) and verify asset integrity (FR-IMG-6). Hard failures (unresolved `FIG` placeholder of any shape, missing asset) mark the job `failed` with a diagnostic report; lint warnings are reported but non-fatal. |
| FR-QA-4 | M | The system SHALL write to the user-specified output directory: `<docname>.md` at the top level (one glance lists all documents), plus per-document `<docname>/assets/` and `<docname>/conversion-report.md` (per-page coverage scores + floor scores, `needs_review` pages, omissions log, token usage, timings, config used — including prompt versions, model identifiers, DPI — the per-figure diagram conversion summary, and the overall conversion rate). Re-converting the same document replaces its own files only. |
| FR-QA-6 | C | HTML export (`<docname>.html`, GFM-rendered) as a post-processing option; default off. Not implemented. |

### 4.8 Web UI (UI)

| ID | Priority | Requirement |
|---|---|---|
| FR-UI-1 | M | Upload screen: drag-and-drop or file picker (PDF only), output-directory input with validation feedback, server-side folder browser, conversion options (DPI, verification threshold, TOC on/off, `<details>` descriptions on/off, diagrams→Mermaid on/off), Ollama Cloud privacy notice. |
| FR-UI-2 | M | Progress screen: overall progress bar, current stage, page N/M, per-page status grid (pending/ocr/transcribed/verified/needs_review/failed), live log tail, and streaming Markdown preview of completed pages — updated via WebSocket with reconnect replay. |
| FR-UI-3 | M | Controls: Cancel, Pause/Resume, plus Restart-from-checkpoint for stalled jobs. |
| FR-UI-4 | M | Completion screen: output path, Markdown + report downloads (served in-page, failures surface inline), conversion-report summary, and the list of `needs_review` pages with coverage scores. |
| FR-UI-5 | S | Job history list (FR-JOB-7) with re-open of past jobs. |
| FR-UI-6 | M | Setup/health panel: shows connectivity status of local Ollama (+ `glm-ocr` present), Ollama Cloud (key valid), PostgreSQL, disk — with actionable errors. |

---

## 5. External Interface Requirements

### 5.1 REST API (FastAPI, `/api/v1`)

| Method & Path | Purpose |
|---|---|
| `POST /jobs` | Create job (multipart: `file`, `output_dir`, options JSON). Returns `job_id`. 409 if a job is running. |
| `GET /jobs` | List jobs (history, newest first, 50 max). |
| `GET /jobs/{id}` | Job detail: status, stage, page progress, config, timings, live flag. |
| `GET /jobs/{id}/pages` | Per-page statuses, coverage scores, needs_review flags. |
| `GET /jobs/{id}/preview` | Assembled-so-far Markdown (terminal pages only). |
| `GET /jobs/{id}/report` | Conversion report (after completion/failure; 404 while running). |
| `GET /jobs/{id}/events` | Audit event rows for the job. |
| `GET /jobs/{id}/artifact/document` | Download the final `.md` (recorded path, else conventional locations). |
| `GET /jobs/{id}/artifact/report` | Download `conversion-report.md` (recorded path, else conventional locations). |
| `POST /jobs/{id}/restart` | Resume a stalled job (paused/cancelled/failed/crashed) from its checkpoints using its own snapshotted options. 202 on accept. |
| `POST /jobs/{id}/cancel` | Cancel running job. |
| `POST /jobs/{id}/pause` · `POST /jobs/{id}/resume` | Pause/resume a live job. Terminal or unknown-control states explain the way back (restart hint). |
| `GET /browse?path=` | Server-side folder browser (localhost operator's own disk; directories only, capped). |
| `GET /health` | Component health: ollama_local, local_models, ollama_cloud, postgres, disk, plus the running job id. |
| `WS /ws/jobs/{id}` | Progress event stream (see 5.2), with replay buffer for reconnects. Clean shutdown: server-side cancellation exits quietly. |

All errors return RFC 9457 problem-details JSON.

### 5.2 WebSocket Events

```json
{ "event": "stage_changed", "job_id": "…", "stage": "transcribing" }
{ "event": "page_update", "page": 42, "total": 310, "status": "verified", "coverage": 98 }
{ "event": "page_markdown", "page": 42, "markdown": "…" }
{ "event": "log", "level": "info", "message": "…" }
{ "event": "job_finished", "status": "completed", "output_path": "…" }
{ "event": "error", "message": "invalid job id: …" }
```

### 5.3 Model Interfaces (Ollama)

| Model | Endpoint | Notes |
|---|---|---|
| `glm-5.3-flash` (Cloud variant; locally referenced as `glm-5.3-flash:cloud`) | **Ollama Cloud** — dedicated client via `OLLAMA_CLOUD_URL` + `Authorization: Bearer OLLAMA_API_KEY`; never the local daemon | Multimodal chat; images passed as base64; thinking effort configurable (default `low` for transcription, `high` for the diagram conversion/verification calls). **Does not support structured outputs** — responses use the textual envelope contract (Appendix A.2/A.3/A.5). Direct Cloud listing uses the bare name `glm-5.3-flash` (verified via `GET https://ollama.com/api/tags`); the `:cloud` suffix is the local-daemon routing form. The adapter accepts both forms and resolves via a preflight `/api/tags` check. |
| `glm-ocr` | Local daemon `OLLAMA_LOCAL_URL` (`http://localhost:11434`) | **Local only — no cloud variant exists.** Preflight verifies it is pulled and responding. Wall-clock deadline caps degenerate token-trickle loops. |

---

## 6. Data Requirements

### 6.1 PostgreSQL Schema (logical)

```
jobs
  id (uuid, pk) · filename · file_sha256 · page_count · status · stage
  output_dir · options (jsonb, incl. results.artifacts) · pdf_metadata (jsonb)
  pipeline_version · created_at · started_at · finished_at · error (jsonb, nullable)

pages
  id (pk) · job_id (fk) · page_number · status
  render_dpi · ocr_output (jsonb) · markdown (text)
  coverage_score (int) · retries (int) · needs_review (bool)
  omissions (jsonb: figures · furniture · notes · floor_score)
  token_usage (jsonb) · timings (jsonb)
  source_page_hash · render_hash · ocr_hash · markdown_hash
  UNIQUE (job_id, page_number)

images
  id (pk) · job_id (fk) · page_number · asset_path · source ("native"|"crop")
  bbox (jsonb) · alt_text · caption · referenced (bool)
  mermaid (text, nullable) · diagram_type (varchar, nullable)
  conversion_status ("mermaid"|"table"|"image", nullable) · confidence (int, nullable)

events   -- audit/log trail
  id (pk) · job_id (fk) · ts · level · stage · message (text)
```

Migrations managed by Alembic (single head revision). Checkpoint = the `pages` row commit; resume = first page whose status ≠ `verified`/`needs_review`.

**Verified-page immutability:** a page whose checkpoint reached `verified`/`needs_review` is immutable — the pipeline SHALL NOT re-run it because a later page failed or the job was restarted. Re-transcription happens only via an explicit recovery/invalidation operation (user action or config-change job restart), never as an implicit side effect of resume.

**Deterministic replay hashes:** `source_page_hash` pins the PDF content streams of that page; `render`/`ocr`/`markdown` hashes pin each artifact; the job's `pipeline_version` hashes prompt versions + model identifiers + DPI + coverage threshold + coverage floor + diagram config — any input change yields a new version, never a silent resume mix.

### 6.2 Filesystem Layout

```
<workspace>/jobs/{job_id}/          # internal, temp
  source.pdf
  renders/page-001.png …
  figures/page-001-fig-01.png   # per-figure crops for conversion
  assets/  (staging before final copy)

<user output dir>/                  # deliverable (many docs share one dir)
  <docname>.md                      # top level: one glance lists all documents
  <docname>/assets/page-001-img-01.png …
  <docname>/conversion-report.md
```

Image links inside `<docname>.md` are `<docname>/assets/…` (relative, portable). Re-converting the same document replaces only its own `.md` + folder.

---

## 7. Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-1 | Fidelity | **Primary NFR.** ≥ 95% content coverage per page (agent-verified) AND a deterministic token-recall floor ≥ 80% (configurable), zero silently dropped figures, zero silently dropped pages. Every figure is reinterpreted as Mermaid when faithful, otherwise retained as an image or grounded table. Correctness beats latency in every trade-off. Any remaining `<!--FIG:` marker of any shape fails the job loudly. |
| NFR-2 | Reliability | A crash/restart at any point loses at most one page of progress; resume completes without duplicating content. |
| NFR-3 | Performance | No hard time limit per job (explicitly accepted). Indicative target: ≤ 90 s/page average at defaults on a typical connection; OCR look-ahead (FR-OCR-4) keeps the cloud agent the bottleneck. Diagram conversion adds one Cloud call per figure plus one verification call when enabled. (Local OCR on 8 GB-VRAM machines remains the slow-path bottleneck: degenerate pages can take tens of minutes through the DPI-fallback ladder.) |
| NFR-4 | Capacity | Support PDFs up to 1,000 pages / 500 MB (configurable). |
| NFR-5 | Security | Server binds `127.0.0.1` by default. `OLLAMA_API_KEY` and DB credentials only via env/.env (`SecretStr`, never logged, never stored in DB). Path traversal protection on `output_dir` and doc names. |
| NFR-6 | Privacy | User is informed (UI notice) that page images **and cropped figure regions** are sent to Ollama Cloud for the agent and diagram stages. OCR remains fully local. |
| NFR-7 | Portability | Windows 11 native only with native local installs: Python 3.14, Ollama, PostgreSQL 18.x. All dependencies run as native Windows services/processes. No Node/mermaid-cli requirement (verification is vision-model based). |
| NFR-8 | Cloud-readiness | No architectural blockers to future cloud deployment: 12-factor config, stateless API layer, all state in Postgres/filesystem abstractions, job worker separable into its own process later. |
| NFR-9 | Observability | Structured logging (JSON option), per-stage timings, token/cost accounting in the report. Per-figure conversion status/type/confidence in the report. |
| NFR-10 | Maintainability | Type-hinted code (mypy strict clean), pipeline stages behind interfaces (`OcrEngine`, `TranscriptionAgent`, `DiagramConverter`) so models can be swapped by config. |
| NFR-11 | Test taxonomy | Four test tiers, explicitly separated: **Unit** (no network/DB/Ollama), **Integration** (real Postgres/local Ollama, cloud model mocked), **Live model** (real GLM-OCR + GLM-5.3-Flash Cloud; tagged `@pytest.mark.live`, excluded by default — never run routinely, they burn quota), **Acceptance** (real corpus + human review per §10). |
| NFR-12 | Model-contract guardrails | Envelope round-trip, Mermaid validation, and placeholder-resolution behavior are covered by unit/integration tests; drift detection relies on preflight checks plus live smoke tests. |
| NFR-14 | Measurable claims | Any published quality claim about conversion SHALL come from a reproducible, committed corpus run; the previous retrieval benchmark harness was retired with the embeddings pipeline. |

---

## 8. Error Handling & Recovery

| Failure | Behavior |
|---|---|
| Ollama Cloud unreachable / auth failure | Preflight check fails fast with actionable message. Mid-job: retry w/ backoff (3×); then **pause** job (not fail) — resumable when connectivity returns. |
| Model returns malformed output (envelope structure missing/unbalanced, invalid payload JSON, truncation) | Detect via structural validation of the envelope (Appendix A.2); retry with corrective instruction (hard cap, default 2 attempts total); after cap → preserve raw response in the page record, flag `needs_review`. Raw responses are never silently discarded. |
| Literal `idx` figure token | Treated as an unresolved figure: integrity hard-fails the job (FR-IMG-6/FR-QA-3). The prompt forbids emitting it (Appendix A.2). |
| Local `glm-ocr` missing/down | Preflight fails with `ollama pull glm-ocr` guidance. Mid-job: pause, resumable. Degenerate OCR loops are deadline-capped with DPI fallback. |
| Rate limit (429) from Ollama Cloud | Honor `Retry-After` (capped), backoff, continue; surface "throttled" state in UI. |
| Page verification below threshold after max retries | Mark page `needs_review`, include best attempt, continue. Listed prominently in report + UI review screen. Never blocks the job. |
| Coverage floor fails on an otherwise-passing page | Treated identically to a verdict failure: retry with corrective instruction; after cap, `needs_review` with the floor score and a "coverage floor failed" note in the page record. |
| Diagram conversion fails / verifier rejects the Mermaid | Fall back by tier: grounded data table + image for data charts, else image + alt-text + caption. The figure is never dropped; the report records the fallback reason. |
| Diagram conversion envelope malformed | Treat as non-convertible (image fallback); the malformed response is logged. Never fails the job. |
| Workspace cleanup fails with `PermissionError` (Windows file lock: open handle or Defender scan) | Log warning, retry with exponential backoff up to `MAX_CLEANUP_RETRIES`; if still locked, mark workspace for cleanup on next app start. Never fails a completed job. |
| Output dir becomes unwritable / disk full | Pause job with explicit error; resume after user fixes. |
| Postgres unavailable at startup | App starts; startup scan is skipped with a warning and the health panel shows the problem (degraded mode). |
| Encrypted/corrupt PDF | Preflight failure with reason. |
| App crash / power loss | On restart, incomplete jobs are logged with their restart command and offered for resume (FR-JOB-5). |
| Server shutdown (Ctrl+C) with open WebSockets | Sockets close with status 1001; in-flight waits exit quietly — no error spam. Background job tasks are cancelled. |

---

## 9. Constraints & Assumptions

**Constraints**

- C-1: Python 3.14; all dependencies must support it.
- C-2: `glm-ocr` has **no Ollama Cloud variant** — must run on the local Ollama daemon.
- C-3: One PDF/job at a time (single worker).
- C-4: Output must be GFM that renders correctly on github.com without extensions beyond GFM + `$…$` math + native Mermaid fenced blocks.
- C-5: PostgreSQL is the only persistence technology.

**Assumptions**

- A-1: User has a valid Ollama Cloud API key (Pro) with sufficient quota for image-heavy requests.
- A-1a: Cloud model requests are authenticated exclusively via `OLLAMA_API_KEY` against `OLLAMA_CLOUD_URL`; the interactive `ollama signin` state of the machine is never relied upon by the application.
- A-2: Source PDFs are primarily English or languages supported by both GLM models.
- A-3: The local machine has enough disk for rendered pages (~0.5–1.5 MB/page at 200 DPI).

---

## 10. Acceptance Criteria

The program works when the following hold on a test corpus of ≥ 5 PDFs (a scanned book chapter, a technical paper with formulas + multi-page tables, a slide-deck-style PDF, an image-heavy report, a 2-column academic paper):

1. **AC-1 Completeness:** Every page of every corpus PDF appears in the output; the report lists zero silently skipped pages; every figure in the source is present as an extracted asset with alt-text (manual audit).
2. **AC-2 Fidelity:** For 3 randomly sampled pages per document, human review confirms verbatim text (no paraphrase beyond logged typo corrections), correct heading levels, correct table structure, correct reading order.
3. **AC-3 GitHub rendering:** The output `.md` + per-document folder pushed to a GitHub repo renders with no broken images, no malformed tables, math rendered.
4. **AC-4 Resume:** Killing the process mid-job (during transcription) and resuming produces output byte-identical in structure to an uninterrupted run (no duplicated/missing pages).
5. **AC-5 Isolation of failures:** Forcing one page below the coverage threshold results in a completed job with that page flagged `needs_review` — not a failed job.
6. **AC-6 Health checks:** With Ollama stopped, or `glm-ocr` not pulled, or the Cloud key invalid, the UI health panel identifies the exact problem before a job can start.
7. **AC-7 Report:** `conversion-report.md` contains per-page coverage, floor score, route, retries, omissions log, token usage totals, and wall-clock timings.
8. **AC-8 Diagram reinterpretation (evaluated):** On the corpus, every flagged figure is either emitted as validated+verified Mermaid, or falls back to a grounded data table / image; the conversion report lists per-figure status and an overall conversion rate; no figure is silently dropped.

---

## 11. Out of Scope

Multi-user support, PDF editing, translation/summarization modes, DOCX/EPUB input, fine-tuning models. Batch queues, library search UI, HTML export, and Mermaid rendering/visual-diffing (no Node dependency) are not part of the program.

---

## 12. Glossary

| Term | Definition |
|---|---|
| **GFM** | GitHub-Flavored Markdown — the Markdown dialect rendered by github.com. |
| **Agent / transcription agent** | GLM-5.3-Flash operating in a loop with vision input, tool results (OCR), rolling context, and self-verification. |
| **Diagram→Mermaid conversion** | The per-figure stage that crops a figure, grounds it in native text, asks the Cloud model to reinterpret it as Mermaid, validates the type/allowlist, and (by default) vision-verifies fidelity. |
| **Mermaid** | Text-based diagram language rendered natively by GitHub in fenced ` ```mermaid ` blocks. |
| **Tiered fallback** | Mermaid → grounded data table (+image) for data charts → image + alt-text/caption. Ensures no figure is ever silently dropped. |
| **Rolling context** | Document outline + last N pages of produced Markdown fed into each page call for continuity. |
| **Coverage score** | Agent-computed 0–100 estimate of how much of the page's content its Markdown captures. |
| **Running furniture** | Repeating headers, footers, and page numbers excluded from body output. |
| **Coverage floor** | Objective lower bound on page completeness: fraction of OCR-reference tokens present in the produced Markdown after deterministic noise stripping. Configurable via `COVERAGE_FLOOR_TOKENS`; unmeasurable references never gate. |
| **`needs_review`** | Page that failed verification after max retries; delivered with a warning instead of blocking the job. |
| **Textual envelope** | The Cloud-model response contract (Appendix A.2/A.3/A.5): content delivered between strict literal markers (e.g. `<<<MARKDOWN>>>` … `<<<END_MARKDOWN>>>`) and parsed deterministically by the application. Required because Ollama Cloud does not support structured outputs. |
| **Fidelity** | Text fidelity and visual/layout fidelity are **equally important**. Never modify source content merely to make Markdown cleaner — normalization may fix Markdown *syntax* but must never alter transcribed source semantics or wording (extends FR-AGT-6). |
| **Pipeline version** | Identifier/hash of prompt versions + model names + DPI + coverage threshold + coverage floor + diagram config; recorded in `jobs.options`, per-page hashes, and the conversion report. |
| **Born-digital** | PDF with a real text layer (vs scanned image PDF). |

---

## Appendix A — Prompt Contracts

*(Normative interfaces; exact wording is an implementation detail, structure is not. Prompt identifiers in use: `transcription-v2`, `verification-v2`, `diagram-v1`, `diagram-verification-v1`.)*

### A.1 GLM-OCR page call (local)

- Input: page PNG + prompt from the model's supported set — `"Text Recognition:"`, `"Table Recognition:"`, `"Formula Recognition:"` per region need.
- Output: raw recognized content, stored as-is in `pages.ocr_output`.

### A.2 Agent transcription call (per page)

**System:** role = expert document transcriber; rules: verbatim with typo-correction allowance, GFM only, no summarizing, every figure → `<!--FIG:page:INDEX:x0,y0,x1,y1-->` placeholder (e.g. `<!--FIG:page:1:100,200,300,400-->` — replace `INDEX` with 1, 2, 3, …; never emit the literal text `idx`) + alt-text draft, continue multi-page constructs from context, exclude running furniture (list it separately), reading order top-to-bottom column-aware.

**User content:** page image · GLM-OCR output (labeled "OCR reference — trust for characters, not for structure") · rolling context block · document outline so far · **the exact rendered page image dimensions in pixels** (so bbox values are anchored, per FR-AGT-3).

**Output format constraint (normative):** Ollama Cloud does **not** support structured outputs. The model response therefore MUST use the **textual envelope** contract below; the application parses it deterministically (string/regex section extraction — never `format=<JSON schema>` or Pydantic-enforced structured output on Cloud calls). The `FIGURES`/`FURNITURE` payloads inside the envelope are JSON *text* validated by the application after extraction, so one unescaped quote in Markdown cannot invalidate the whole response.

```text
<<<MARKDOWN>>>
[the page's verbatim GFM — may contain anything, including text that
 resembles envelope markers; parser anchors on line-start markers]
<<<END_MARKDOWN>>>
<<<FIGURES>>>
[JSON array: {index, bbox [x0,y0,x1,y1 absolute pixels of the rendered
 PNG per FR-AGT-3], alt, caption}]
<<<END_FIGURES>>>
<<<FURNITURE>>>
[JSON: header / footer / page-number strings, empty objects if none]
<<<END_FURNITURE>>>
<<<NOTES>>>
[continuations, uncertainties [?], typo corrections, decorative-element omissions]
<<<END_NOTES>>>
```

**Parser rules:** markers are matched at line start, exact and case-sensitive; if any required section is missing/unbalanced (structural validation), retry with a corrective instruction; if still malformed after the retry cap, preserve the raw response verbatim in the page record, degrade to `needs_review` — **never silently discard the raw response**.

### A.3 Verification call (per page)

Input: page image + OCR output + candidate Markdown.

Output (same textual-envelope mechanism, payload JSON — see FR-AGT-3a):

```text
<<<VERDICT>>>
{"coverage": 0-100, "misses": ["…"], "structure_issues": ["…"], "verdict": "pass" | "retry"}
<<<END_VERDICT>>>
```

If the JSON payload inside the envelope fails validation → corrective retry (same cap and raw-preservation rule as A.2). Fabrications are reported under `structure_issues` (no schema change needed).

### A.5 Diagram→Mermaid call (per figure)

Input: figure crop + native-text grounding inside the region.

```text
<<<DIAGRAM>>>
{"convertible": true | false, "type": "<mermaid type or empty>", "confidence": 0-100, "description": "…"}
<<<END_DIAGRAM>>>
<<<MERMAID>>>
[the complete Mermaid source, no ``` fences; empty when convertible is false]
<<<END_MERMAID>>>
<<<DATA>>>
{"columns": ["…"], "rows": [["…", "…"]]}
<<<END_DATA>>>
```

Rules: a figure is convertible only when its structure maps to a supported Mermaid type (flowchart, sequence, class, state, ER, gantt, mindmap, timeline, journey, pie, gitGraph, quadrant, xychart, sankey, architecture, radar, kanban); photographs, artwork, maps, logos, and pixel-dependent schematics are not convertible. No node/edge/label/value may be invented. `DATA` carries extracted values for the table fallback on data charts.

### A.6 Diagram verification call (per figure)

Input: figure crop + grounding + candidate Mermaid. Output: the A.3 verdict envelope where `coverage` is reinterpretation fidelity; any fabrication is blocking.

---

## Appendix B — Configuration Reference (`.env` / options)

| Key | Default | Description |
|---|---|---|
| `OLLAMA_LOCAL_URL` | `http://localhost:11434` | Local daemon (glm-ocr) — no auth |
| `OLLAMA_CLOUD_URL` | `https://ollama.com` | Ollama Cloud base URL — Cloud client targets this **directly**; never routed through the local daemon |
| `OLLAMA_API_KEY` | — (required) | Ollama Cloud auth — sent as `Authorization: Bearer` by the Cloud client only |
| `AGENT_MODEL` | `glm-5.3-flash` | Transcription/diagram agent (Cloud). Accepts local-daemon form `glm-5.3-flash:cloud`; adapter resolves via preflight `/api/tags` (§5.3) |
| `OCR_MODEL` | `glm-ocr` | Local OCR model — **never routed to Cloud** |
| `AGENT_TIMEOUT_SECONDS` | `300` | Per-call timeout, Cloud agent |
| `OCR_TIMEOUT_SECONDS` | `120` | Per-call timeout, local OCR |
| `AGENT_MAX_OUTPUT_TOKENS` | — (optional) | Output-token cap for Cloud calls when set |
| `RENDER_DPI` | `200` | Base page render DPI (max auto 400) |
| `ROLLING_CONTEXT_PAGES` | `3` | Prior pages included per call (config-driven, never hard-coded) |
| `COVERAGE_THRESHOLD` | `95` | Min verified coverage per page (agent verdict) |
| `COVERAGE_FLOOR_TOKENS` | `80` | Deterministic floor: OCR-token recall each page must meet (percent, `0` = off) |
| `COVERAGE_FLOOR_MIN_OCR_TOKENS` | `30` | References with fewer tokens are unmeasurable and never gate |
| `MAX_PAGE_RETRIES` | `2` | Verification retries per page |
| `NATIVE_TEXT_FIRST` | `true` | Per-page routing: use a substantial native text layer as the character reference instead of OCR |
| `OCR_ENABLED` | `true` | Master OCR switch; `false` = vision-only for pages without native text, zero `glm-ocr` calls |
| `NATIVE_TEXT_MIN_WORDS` | `20` | Native-layer word floor for reference routing (character fallback for space-less scripts) |
| `MAX_PDF_MB` | `500` | Upload cap |
| `MAX_PDF_PAGES` | `1000` | Page cap |
| `THINKING_EFFORT_TRANSCRIBE` | `low` | GLM-5.3-Flash effort, page calls |
| `THINKING_EFFORT_DIAGRAM` | `high` | GLM-5.3-Flash effort, diagram conversion/verification |
| `DIAGRAM_TO_MERMAID` | `true` | Enable figure→Mermaid reinterpretation (primary capability) |
| `DIAGRAM_ALLOWED_TYPES` | all 17 types | Comma-separated Mermaid type allowlist |
| `DIAGRAM_MIN_CONFIDENCE` | `80` | Minimum converter confidence accepted |
| `DIAGRAM_VERIFY` | `true` | Vision-verify each candidate Mermaid against the crop |
| `DIAGRAM_FALLBACK` | `both` | `image` \| `table` \| `both` (data charts get tables) |
| `DIAGRAM_KEEP_IMAGE` | `true` | Keep original figure under a converted Mermaid |
| `TOC_ENABLED` | `true` | Insert generated TOC |
| `FIG_DETAILS_BLOCKS` | `true` | `<details>` long descriptions for charts |
| `KEEP_WORKSPACE_ON_SUCCESS` | `false` | Retain renders after completion |
| `MAX_CLEANUP_RETRIES` | `3` | Windows file-lock retries (exponential backoff) before workspace cleanup is deferred to next start |
| `DATABASE_URL` | — (required) | PostgreSQL DSN |
| `BIND_HOST` / `PORT` | `127.0.0.1` / `8000` | Server binding |

---

*End of SRS.*
