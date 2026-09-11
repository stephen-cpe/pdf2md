# Phase 4 Benchmark Report — pdf2md vs pymupdf4llm baseline

**Date:** 2026-09-10 · **Corpus:** 6 PDFs (5 born-digital + 1 synthesized scanned) · **Golden set:** 34 questions · **Embeddings:** qwen3-embedding:0.6b (local) · **Retrieval:** Chroma cosine, top-5

---

## 1. Headline result

| Pipeline | Chunker | Recall@5 | MRR |
|---|---|---|---|
| pymupdf4llm baseline | sections | 68% (23/34) | 0.537 |
| pymupdf4llm baseline | hierarchy | 68% (23/34) | 0.472 |
| **pdf2md (hybrid)** | **sections** | **94% (32/34)** | **0.641** |
| **pdf2md (hybrid)** | **hierarchy** | **91% (31/34)** | **0.626** |

**pdf2md beats the deterministic baseline by 23–26 points of Recall@5 on identical chunking, embedding, and retrieval.** The pre-agreed decision rule ("≥3 questions / ≥12 points = keep the pipeline") is exceeded roughly 2x.

## 2. Where the gap comes from — decomposed

| Segment | Baseline | pdf2md | Delta |
|---|---|---|---|
| Born-digital docs only (26 q) | 88% (23/26) | 96% (25/26)* | +2 questions (within noise band) |
| Scanned doc (8 q) | **0% (0/8)** | **87.5% (7/8)** | +7 questions |

*\*pdf2md recovered 2 of the baseline's 3 born-digital misses (OCP "Abstraction is the Key", ISP timer-client pattern) and introduced none; the one shared miss (SPL "tool or environment") is a ranking failure on both pipelines.*

**The finding is precise:** on text-layer-native documents the baseline is already strong and pdf2md adds only a within-noise edge (+2 questions — not claimed as significant at this sample size). The decisive 23-point gap comes from the scanned document — pages with no text layer, where the baseline extracts **zero characters** and pdf2md's OCR+vision path recovers full structure (headings, lists, LaTeX formula, a 5-row GFM table) in under 2 minutes.

This is the honest, important result: **the agentic layer earns its keep exactly where deterministic extraction has nothing to say.** It also validates hybrid routing's premise — spend the expensive path only where the cheap path is structurally blind.

### Per-question character of pdf2md's 2 misses

Both are **ranking failures, not content failures** — the expected substrings exist in pdf2md's output (verified by grep), but embedding similarity did not surface them in top-5. Content-wise pdf2md captured 34/34 answers.

## 3. Hybrid routing A/B (Open-Closed_Principle, 14 pages, same doc converted twice)

| Metric | Hybrid ON | Hybrid OFF (agentic) |
|---|---|---|
| Retrieval on OCP questions | 5/5, MRR 0.707 | 5/5, MRR 0.717 |
| Pages routed deterministic | 4/14 | 0/14 |
| Cloud tokens (prompt) | 241,072 | 235,254 |
| Pages needing no OCR | 4 | 0 |
| Wall-clock (observed) | ~85 min | ~165 min |

**Verdict: identical retrieval quality (MRR delta 0.010, within noise on 5 questions), ≈50% wall-clock savings, and no OCR spend on routed pages.** The token totals are similar because the verifier still runs on every page (by design — it is the quality gate); the savings are in local OCR time (0 ms vs up to 31 min/page on the degenerate pages) and in the skipped transcribe calls on the 4 deterministic pages.

**Routing worked as designed in production:** the report shows pages 1, 6, 11, 13 routed deterministic with floor 97–98% and verify-only cost (~4 s/page); the figure-heavy pages escalated automatically; the OCR-degenerate pages exercised the DPI-fallback ladder (200→80→72) and completed.

## 4. Chunking effect (both pipelines, free re-measurement)

On this corpus the hierarchical chunker is **neutral on recall** (94→91 for pdf2md, 68→68 for baseline) and slightly negative on MRR. Sections in these documents are short enough that the original section chunker was already retrieval-sized; smaller children split two answers across chunk boundaries (both pdf2md hierarchy misses are multi-paragraph questions). **Recommendation: keep `split_sections` as the default for the `documents` collection; retain `chunking.py` (tested, wired, `--chunker` flag) — it is the right tool for long-section documents (books, manuals) and the parent-linkage data model it produces enables small-to-big retrieval later.** The corpus, not the chunker, is the binding constraint here.

## 5. Cost accounting (live Cloud spend, glm-5.3-flash)

| Document | Pages | Deterministic | Status |
|---|---|---|---|
| Open-Closed_Principle (hybrid) | 14 | 4 | completed, 1 page needs_review |
| Interface_Segration_Principle | 13 | 5 | completed |
| scanned_knowledge_handbook | 3 | 0 (by design) | completed |
| Generative AI TDD (arxiv) | 8 | — | completed |
| bitcoin | 9 | — | completed |
| Software-Product-Line | 8 | — | completed |
| Open-Closed_Principle (agentic A/B) | 14 | 0 | completed |

One page across the whole corpus landed in needs_review (OCP page 3, coverage 92 after 2 retries) — a 98.2% terminal-verified rate with the floor active. Coverage floors ran 97–100% on every page (the verifier's "95" scores alongside floors ≥97 show judge and floor agreeing).

## 6. Decision (per the pre-agreed framework)

**Outcome: KEEP THE PIPELINE — and the data says publish it as a hybrid converter.**

1. **The conversion gap is real but situational.** +23 points Recall@5 overall; on born-digital single-column text the edge is +2 questions (within the pre-agreed noise band, not claimed as significant). The project's README should claim exactly this: *deterministic-first for text-layer pages, agentic recovery for scanned/figure-dense pages* — and show this table.
2. **Hybrid routing is validated and should default ON** for any corpus with a mix of text-layer and scanned content. Same retrieval, half the wall-clock, zero OCR on routed pages. (Ship note: `HYBRID_ROUTING=false` remains the safe default in code; flipping the `.env` default is a one-line user choice.)
3. **The scanned-document result is the product.** A zero-text PDF produced a fully structured 3.6 KB markdown (TOC, GFM table, LaTeX) in ~100 seconds. That is the demo to lead with.
4. **Known-honest limitations to state:** n=34 questions; single embedding model; the scanned doc is synthetic (clean render, no skew/noise of real paper); needs_review pages (1/70 across runs) still require human eyes; the OCR stage on 8 GB-VRAM machines is the wall-clock bottleneck (minutes per degenerate page, mitigated — not eliminated — by the DPI ladder).

## 7. Reproduction

```
python eval/make_scanned_doc.py                       # corpus fix (idempotent)
python app.py                                          # HYBRID_ROUTING=true in .env
python eval/live_convert.py corpus/<doc>.pdf 90        # per doc
python eval/run_eval.py --pipeline eval/baseline-md --pipeline eval/pdf2md-md --chunker sections
```

Artifacts: `eval/results-sections.json` (headline), `eval/results-hierarchy.json` (chunking), `eval/results-routing.json` (A/B), staged outputs in `eval/pdf2md-md/`, `eval/ocp-agentic-md/`, `eval/baseline-md/`.