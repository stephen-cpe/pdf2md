"""Retrieval evaluation harness — golden set + Recall@k + MRR.

Compares RAG retrieval quality across conversion pipelines (e.g.
pymupdf4llm baseline vs pdf2md agentic output) under IDENTICAL
chunking, embedding, and retrieval — isolating the effect of the
document representation itself.

Usage:
    python eval/run_eval.py --pipeline baseline-md --top-k 5
    python eval/run_eval.py --pipeline baseline-md --pipeline <pdf2md-output-dir> --report eval/results.json

Layout conventions:
    eval/goldenset.json          ground-truth questions (committed)
    eval/baseline-md/*.md        pymupdf4llm extraction (generated)
    <pipeline dir>/*.md          any other pipeline's markdown output

Metrics (per question, averaged per pipeline):
    hit@k   — any retrieved chunk contains an expected substring
    MRR@k   — 1/rank of the first hitting chunk
Chunks come from the project's own split_sections (heading-aware) so
the comparison reflects the real ingestion path, not a toy chunker.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chroma import ChromaStore
from src.config import load_settings
from src.pipeline.chunking import chunk_children
from src.pipeline.report import split_sections

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_SET = EVAL_DIR / "goldenset.json"
DEFAULT_BASELINE = EVAL_DIR / "baseline-md"
RESULTS_FILE = EVAL_DIR / "results.json"
COLLECTION_PREFIX = "eval_"


def load_golden_set_from(path: Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [q for q in data["questions"] if "question" in q and "expected" in q]


def chunk_document(markdown_path: Path, chunker: str) -> list[tuple[str, str]]:
    """(heading/heading_path, text) chunks for one markdown document."""
    text = markdown_path.read_text(encoding="utf-8")
    if chunker == "hierarchy":
        return [(child.heading_path_str, child.text) for child in chunk_children(text)]
    return split_sections(text)


def index_pipeline(store: ChromaStore, pipeline_dir: Path, pipeline_name: str, chunker: str) -> int:
    """Embed every chunk of every .md in pipeline_dir; returns chunk count."""
    collection = f"{COLLECTION_PREFIX}{pipeline_name}_{chunker}"
    store.delete_collection(collection)
    ids: list[str] = []
    docs: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for md in sorted(pipeline_dir.glob("*.md")):
        stem = md.stem
        for pos, (heading, text) in enumerate(chunk_document(md, chunker)):
            ids.append(f"{stem}:{pos:04d}")
            docs.append(text)
            metadatas.append({"doc": stem, "heading": heading, "chunk": pos})
    if docs:
        store.add_texts(collection, ids, docs, metadatas)
    return len(docs)


def retrieve(
    store: ChromaStore,
    pipeline_name: str,
    question: str,
    top_k: int,
    chunker: str,
) -> list[dict[str, Any]]:
    """Top-k chunks for a question; raw Chroma result normalized."""
    result = store.query(f"{COLLECTION_PREFIX}{pipeline_name}_{chunker}", question, n_results=top_k)
    metadatas = result.get("metadatas", [[]])[0]
    documents = result.get("documents", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [
        {
            "doc": m.get("doc", ""),
            "heading": m.get("heading", ""),
            "text": d,
            "distance": dist,
        }
        for m, d, dist in zip(metadatas, documents, distances)
    ]


def question_hit(rank: int, retrieved: list[dict[str, Any]], expected: list[str]) -> int | None:
    """Rank (1-based) of the first chunk containing any expected substring, else None."""
    lowered = [e.lower() for e in expected]
    for pos, chunk in enumerate(retrieved, start=1):
        text = chunk["text"].lower()
        if any(e in text for e in lowered):
            return pos
    return None


def eval_pipeline(
    store: ChromaStore,
    pipeline_dir: Path,
    questions: list[dict[str, Any]],
    top_k: int,
    chunker: str,
) -> dict[str, Any]:
    """Index one pipeline's outputs and score all questions against it."""
    pipeline_name = pipeline_dir.name
    t0 = time.perf_counter()
    chunks = index_pipeline(store, pipeline_dir, pipeline_name, chunker)
    index_seconds = time.perf_counter() - t0
    per_question: list[dict[str, Any]] = []
    hits = 0
    reciprocal_ranks: list[float] = []
    for q in questions:
        retrieved = retrieve(store, pipeline_name, q["question"], top_k, chunker)
        rank = question_hit(0, retrieved, q["expected"])
        hit = rank is not None
        hits += hit
        if hit:
            reciprocal_ranks.append(1.0 / rank)
        per_question.append(
            {
                "doc": q["doc"],
                "question": q["question"],
                "hit": hit,
                "first_hit_rank": rank,
                "top_doc": retrieved[0]["doc"] if retrieved else None,
                "top_heading": retrieved[0]["heading"] if retrieved else None,
            }
        )
    return {
        "pipeline": pipeline_name,
        "dir": str(pipeline_dir),
        "chunker": chunker,
        "chunks_indexed": chunks,
        "questions": len(questions),
        "hit_at_k": hits,
        "recall_at_k": round(hits / len(questions), 4) if questions else 0.0,
        "mrr": round(sum(reciprocal_ranks) / len(questions), 4) if questions else 0.0,
        "index_seconds": round(index_seconds, 1),
        "top_k": top_k,
        "per_question": per_question,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retrieval evaluation for pdf2md")
    parser.add_argument(
        "--pipeline",
        action="append",
        required=True,
        help="Directory of .md outputs to evaluate (repeatable, e.g. --pipeline eval/baseline-md)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--chunker",
        choices=["sections", "hierarchy"],
        default="sections",
        help="sections = one chunk per heading section (original); "
        "hierarchy = paragraph-atomic children with heading paths",
    )
    parser.add_argument(
        "--golden-set", type=Path, default=GOLDEN_SET, help="Path to goldenset.json"
    )
    parser.add_argument("--report", type=Path, default=RESULTS_FILE)
    args = parser.parse_args()

    golden_path = args.golden_set
    questions = load_golden_set_from(golden_path)
    if not questions:
        sys.exit(f"no questions found in {golden_path}")
    print(f"golden set: {len(questions)} questions, top_k={args.top_k}")

    settings = load_settings()
    store = ChromaStore(
        str(settings.CHROMA_PATH),
        settings.EMBED_MODEL,
        ollama_url=settings.OLLAMA_LOCAL_URL,
    )

    results = []
    for pipeline_arg in args.pipeline:
        pipeline_dir = Path(pipeline_arg).resolve()
        if not pipeline_dir.is_dir():
            sys.exit(f"pipeline dir not found: {pipeline_dir}")
        md_files = sorted(pipeline_dir.glob("*.md"))
        if not md_files:
            sys.exit(f"no .md files in {pipeline_dir}")
        needed = {q["doc"] for q in questions}
        present = {f.stem for f in md_files}
        if missing := needed - present:
            print(f"warning: {pipeline_dir.name} is missing docs: {sorted(missing)}")

    for pipeline_arg in args.pipeline:
        pipeline_dir = Path(pipeline_arg).resolve()
        print(f"indexing {pipeline_dir} ({args.chunker}) …", flush=True)
        result = eval_pipeline(store, pipeline_dir, questions, args.top_k, args.chunker)
        results.append(result)
        print(
            f"  {result['pipeline']}: {result['chunks_indexed']} chunks, "
            f"hit@{args.top_k}={result['hit_at_k']}/{result['questions']} "
            f"(recall {result['recall_at_k']:.0%}), MRR={result['mrr']}"
        )

    report = {
        "golden_set": str(golden_path),
        "top_k": args.top_k,
        "chunker": args.chunker,
        "embed_model": settings.EMBED_MODEL,
        "results": results,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report written: {args.report}")

    if len(results) > 1:
        print("\npipeline comparison (recall/MRR):")
        for r in sorted(results, key=lambda r: r["mrr"], reverse=True):
            print(
                f"  {r['pipeline']:30s} recall@{args.top_k}={r['recall_at_k']:.0%}  MRR={r['mrr']}"
            )


if __name__ == "__main__":
    main()
