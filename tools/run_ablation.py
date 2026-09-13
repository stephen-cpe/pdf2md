"""OCR ablation runner: convert each corpus PDF twice — with and without glm-ocr.

Mirrors `POST /api/v1/jobs` exactly (same settings/engine/workspace, same
JobOptions snapshot + pipeline_version + repo.create_job + driver.run_job),
except it runs in-process and sequentially so one invocation produces both
arms with identical config apart from the `ocr_enabled` override.

Usage (pilot first, as agreed):
    venv\\Scripts\\python tools\\run_ablation.py --docs Liskov_Substitution_Principle.pdf
    venv\\Scripts\\python tools\\run_ablation.py --docs A.pdf,B.pdf --arms without
    venv\\Scripts\\python tools\\run_ablation.py --all

Outputs land in output_glm_ocr/ and output_no_glm_ocr/. A manifest JSON with
job ids, options, versions, and timings is written to tools/ after every run.
"""

import asyncio
import datetime
import hashlib
import json
import sys
import uuid
import warnings
from pathlib import Path

# Required first on Windows (asyncpg stability); same rule as app.py.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.app import safe_doc_stem
from src.config import load_settings
from src.db import engine as engine_mod
from src.db import repo
from src.logging import configure_logging
from src.pdf import preflight
from src.pipeline import driver as driver_mod
from src.pipeline.control import JobControl
from src.pipeline.driver import JobOptions
from src.pipeline.prompts import PROMPT_VERSIONS, compute_pipeline_version
from src.workspace import Workspace

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
OUT_WITH = ROOT / "output_glm_ocr"
OUT_WITHOUT = ROOT / "output_no_glm_ocr"
MANIFEST = ROOT / "tools" / "ablation_manifest.json"


def _frozen_options_snapshot(settings, job_options: JobOptions) -> dict:
    return {
        "agent_model": settings.AGENT_MODEL,
        "ocr_model": settings.OCR_MODEL,
        "render_dpi": job_options.render_dpi,
        "rolling_context_pages": job_options.rolling_context_pages,
        "coverage_threshold": job_options.coverage_threshold,
        "coverage_floor": job_options.coverage_floor,
        "max_page_retries": job_options.max_page_retries,
        "thinking_transcribe": job_options.thinking_transcribe,
        "thinking_diagram": job_options.thinking_diagram,
        "native_text_first": job_options.native_text_first,
        "native_text_min_words": job_options.native_text_min_words,
        "ocr_enabled": job_options.ocr_enabled,
        "toc_enabled": job_options.toc_enabled,
        "fig_details": job_options.fig_details,
        "diagram_to_mermaid": job_options.diagram_to_mermaid,
        "diagram_min_confidence": job_options.diagram_min_confidence,
        "diagram_verify": job_options.diagram_verify,
        "diagram_fallback": job_options.diagram_fallback,
        "diagram_keep_image": job_options.diagram_keep_image,
        "prompts": dict(PROMPT_VERSIONS),
    }


def _fingerprint(settings, job_options: JobOptions) -> str:
    return compute_pipeline_version(
        agent_model=settings.AGENT_MODEL,
        ocr_model=settings.OCR_MODEL,
        render_dpi=job_options.render_dpi,
        rolling_context_pages=job_options.rolling_context_pages,
        coverage_threshold=job_options.coverage_threshold,
        coverage_floor=job_options.coverage_floor,
        max_page_retries=job_options.max_page_retries,
        thinking_transcribe=job_options.thinking_transcribe,
        thinking_diagram=job_options.thinking_diagram,
        native_text_first=job_options.native_text_first,
        native_text_min_words=job_options.native_text_min_words,
        ocr_enabled=job_options.ocr_enabled,
        toc_enabled=job_options.toc_enabled,
        fig_details=job_options.fig_details,
        diagram_to_mermaid=job_options.diagram_to_mermaid,
        diagram_min_confidence=job_options.diagram_min_confidence,
        diagram_verify=job_options.diagram_verify,
        diagram_fallback=job_options.diagram_fallback,
        diagram_keep_image=job_options.diagram_keep_image,
    )


async def _run_one(factory, settings, workspace, pdf: Path, out_dir: Path, ocr_enabled: bool) -> dict:
    """Convert one PDF under one arm; returns the manifest record."""
    job_options = JobOptions.from_settings(settings, {"ocr_enabled": ocr_enabled})
    content = pdf.read_bytes()
    if len(content) > settings.MAX_PDF_MB * 1024 * 1024:
        raise SystemExit(f"{pdf.name}: exceeds MAX_PDF_MB")
    job_id = uuid.uuid4()
    job_dir = workspace.job_dir(str(job_id))
    source = job_dir / f"{safe_doc_stem(pdf.name)}.pdf"
    source.write_bytes(content)
    pre = preflight(source, max_mb=settings.MAX_PDF_MB, max_pages=settings.MAX_PDF_PAGES)
    if not pre.valid:
        raise SystemExit(f"{pdf.name}: preflight failed: {pre.error}")
    out = Workspace.validate_output_dir(str(out_dir), create=True)
    snapshot = _frozen_options_snapshot(settings, job_options)
    version = _fingerprint(settings, job_options)
    async with factory() as session:
        await repo.create_job(
            session,
            filename=pdf.name,
            file_sha256=hashlib.sha256(content).hexdigest(),
            page_count=pre.page_count,
            output_dir=str(out),
            options=snapshot,
            pdf_metadata=pre.metadata,
            pipeline_version=version,
            job_id=job_id,
        )
    events: list[dict] = []

    async def _emit(event: dict) -> None:
        events.append({"t": datetime.datetime.now(datetime.UTC).isoformat(), **event})

    started = datetime.datetime.now(datetime.UTC)
    status = await driver_mod.run_job(
        factory=factory,
        settings=settings,
        workspace=workspace,
        job_id=job_id,
        pdf_path=source,
        output_dir=out,
        options=job_options,
        control=JobControl(),
        emit=_emit,
    )
    finished = datetime.datetime.now(datetime.UTC)
    try:
        rel_out = str(out.relative_to(ROOT))
    except ValueError:
        rel_out = str(out)
    return {
        "pdf": pdf.name,
        "arm": "with-ocr" if ocr_enabled else "without-ocr",
        "ocr_enabled": ocr_enabled,
        "job_id": str(job_id),
        "status": status,
        "pages": pre.page_count,
        "wall_seconds": round((finished - started).total_seconds(), 1),
        "pipeline_version": version,
        "events": len(events),
        "output_dir": rel_out,
    }


async def _main(docs: list[str], arms: list[bool]) -> None:
    configure_logging()
    settings = load_settings()
    engine = engine_mod.create_engine(settings.DATABASE_URL.get_secret_value())
    factory = engine_mod.session_factory(engine)
    workspace = Workspace(ROOT / "workspace", max_retries=settings.MAX_CLEANUP_RETRIES)
    workspace.cleanup_deferred()
    manifest: list[dict] = []
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    try:
        for name in docs:
            pdf = CORPUS / name
            if not pdf.is_file():
                print(f"SKIP (missing): {name}")
                continue
            for ocr_enabled in arms:
                out_dir = OUT_WITH if ocr_enabled else OUT_WITHOUT
                arm = "with-ocr" if ocr_enabled else "without-ocr"
                print(f"=== {name} [{arm}] ===", flush=True)
                try:
                    record = await _run_one(factory, settings, workspace, pdf, out_dir, ocr_enabled)
                except Exception as exc:  # noqa: BLE001 - record and continue the matrix
                    record = {
                        "pdf": name,
                        "arm": arm,
                        "ocr_enabled": ocr_enabled,
                        "job_id": None,
                        "status": f"runner-error: {exc}",
                        "pages": None,
                        "wall_seconds": None,
                        "pipeline_version": None,
                        "events": 0,
                        "output_dir": str(out_dir),
                    }
                print(f"--- {name} [{arm}]: {record['status']} in {record['wall_seconds']}s")
                manifest.append(record)
                MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    finally:
        await engine.dispose()
    print(f"manifest: {MANIFEST} ({len(manifest)} records)")


def _parse_args(argv: list[str]) -> tuple[list[str], list[bool]]:
    docs: list[str] | None = None
    arms = [True, False]
    i = 0
    while i < len(argv):
        if argv[i] == "--docs" and i + 1 < len(argv):
            docs = [d.strip() for d in argv[i + 1].split(",") if d.strip()]
            i += 2
        elif argv[i] == "--arms" and i + 1 < len(argv):
            want = {a.strip().lower() for a in argv[i + 1].split(",")}
            arms = []
            if "with" in want:
                arms.append(True)
            if "without" in want:
                arms.append(False)
            if not arms:
                raise SystemExit("--arms must list with and/or without")
            i += 2
        elif argv[i] == "--all":
            docs = None
            i += 1
        elif argv[i] in ("-h", "--help"):
            print(__doc__)
            raise SystemExit(0)
        else:
            raise SystemExit(f"unknown arg: {argv[i]} (see --help)")
    if docs is None:
        docs = sorted(p.name for p in CORPUS.glob("*.pdf"))
    return docs, arms


if __name__ == "__main__":
    _docs, _arms = _parse_args(sys.argv[1:])
    print(f"docs={_docs}")
    print(f"arms={['with-ocr' if a else 'without-ocr' for a in _arms]}")
    asyncio.run(_main(_docs, _arms))
