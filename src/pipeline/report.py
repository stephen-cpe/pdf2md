"""Output writer + conversion report.

Report carries every AC-7 field: per-page coverage/retries/omissions/tokens/
timings/telemetry, needs_review list, token totals, wall-clock, the full
config snapshot (models, DPI, prompt versions, pipeline_version), and the
diagram→Mermaid conversion summary (the primary capability).
"""

import datetime
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.db import repo
from src.db.models import Image, Job, Page
from src.pipeline.gfm import github_math_safe


@dataclass(frozen=True)
class Deliverable:
    """Paths written by write_output."""

    markdown_path: Path
    assets_dir: Path
    report_path: Path


def _wall_clock(job: Job) -> str:
    if job.started_at is None:
        return "not started"
    end = job.finished_at or datetime.datetime.now(datetime.UTC)
    return f"{(end - job.started_at).total_seconds():.1f}s"


def build_report(
    job: Job,
    pages: list[Page],
    *,
    pipeline_version: str,
    prompt_versions: dict[str, str],
    furniture_removed: list[dict[str, object]],
    lint_warnings: list[str],
    images: list[Image] | None = None,
    orphaned_figures: list[dict[str, object]] | None = None,
) -> str:
    """Render conversion-report.md (FR-QA-4, AC-7)."""
    total_prompt = sum((page.token_usage or {}).get("prompt", 0) for page in pages)
    total_completion = sum((page.token_usage or {}).get("completion", 0) for page in pages)
    lines = [
        f"# Conversion report — {job.filename}",
        "",
        f"- job_id: {job.id}",
        f"- status: {job.status.value}",
        f"- pages: {job.page_count}",
        f"- wall-clock: {_wall_clock(job)}",
        f"- tokens: prompt={total_prompt} completion={total_completion}",
        f"- pipeline_version: {pipeline_version}",
        f"- prompts: {prompt_versions}",
        (f"- models: agent={job.options.get('agent_model')} ocr={job.options.get('ocr_model')}"),
        f"- dpi: {job.options.get('render_dpi')}",
        f"- lint warnings: {len(lint_warnings)}",
        "",
        "## Per-page coverage",
        "",
        "| page | status | coverage | retries | dpi | prompt_tok | completion_tok | needs_review |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for page in sorted(pages, key=lambda p: p.page_number):
        usage = page.token_usage or {}
        lines.append(
            f"| {page.page_number} | {page.status.value} | {page.coverage_score} | "
            f"{page.retries} | {page.render_dpi} | {usage.get('prompt', 0)} | "
            f"{usage.get('completion', 0)} | {page.needs_review} |"
        )
    review = [page.page_number for page in pages if page.needs_review]
    lines += ["", "## needs_review pages", "", str(review) if review else "none"]
    lines += ["", "## Diagram → Mermaid conversion", ""]
    lines += _conversion_section(images or [])
    lines += ["", "## Omissions log", ""]
    logged = False
    for page in sorted(pages, key=lambda p: p.page_number):
        omissions = page.omissions or {}
        notes = omissions.get("notes") if isinstance(omissions, dict) else None
        if notes:
            lines.append(f"- page {page.page_number}: {notes}")
            logged = True
    if not logged:
        lines.append("none")
    lines += ["", "## Running furniture stripped", ""]
    if furniture_removed:
        for entry in furniture_removed:
            lines.append(f"- {entry.get('text')!r} (on {entry.get('pages')} pages)")
    else:
        lines.append("none")
    lines += ["", "## Per-page telemetry (FR-AGT-10)", ""]
    for page in sorted(pages, key=lambda p: p.page_number):
        timings = page.timings or {}
        omissions = page.omissions or {}
        floor = omissions.get("floor_score") if isinstance(omissions, dict) else None
        floor_text = "n/a" if floor is None else f"{floor:.0%}"
        lines.append(
            f"- page {page.page_number}: dpi={page.render_dpi} "
            f"image={timings.get('image_width')}x{timings.get('image_height')} "
            f"{timings.get('image_bytes')}B model={timings.get('model')} "
            f"effort={timings.get('thinking_effort')} "
            f"transcribe_ms={timings.get('transcribe_ms')} verify_ms={timings.get('verify_ms')} "
            f"ocr_ms={timings.get('ocr_ms')} retries={page.retries} "
            f"verification_score={page.coverage_score} floor={floor_text} "
            f"hashes={page.source_page_hash}/{page.render_hash}/{page.ocr_hash}/{page.markdown_hash}"
        )
    lines += ["", "## Figures placed without placeholder (appended at page end)", ""]
    if orphaned_figures:
        for entry in orphaned_figures:
            lines.append(
                f"- page {entry.get('page')} index={entry.get('index')} "
                f"alt={str(entry.get('alt'))[:80]!r} reason={entry.get('reason', '')}"
            )
    else:
        lines.append("none")
    lines += ["", "## Lint warnings (non-fatal)", ""]
    lines += [f"- {warning}" for warning in lint_warnings] or ["none"]
    return "\n".join(lines) + "\n"


def _conversion_section(images: list[Image]) -> list[str]:
    """Per-figure conversion table + headline conversion rate."""
    if not images:
        return ["none"]
    mermaid = [img for img in images if img.conversion_status == "mermaid"]
    table = [img for img in images if img.conversion_status == "table"]
    image = [img for img in images if img.conversion_status not in ("mermaid", "table")]
    rate = len(mermaid) / len(images) * 100.0
    out = [
        f"- figures: {len(images)}",
        f"- converted to Mermaid: {len(mermaid)} ({rate:.0f}%)",
        f"- data-table fallbacks: {len(table)}",
        f"- image fallbacks: {len(image)}",
        "",
        "| page | index | type | status | confidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for img in sorted(images, key=lambda i: (i.page_number, i.id or 0)):
        out.append(
            f"| {img.page_number} | {img.id} | {img.diagram_type or ''} | "
            f"{img.conversion_status or 'image'} | {img.confidence if img.confidence is not None else ''} |"
        )
    return out


def _validate_docname(docname: str) -> str:
    """Guard against traversal: docname must be a single safe path segment."""
    if not docname or docname in (".", ".."):
        raise ValueError(f"invalid docname: {docname!r}")
    if "/" in docname or "\\" in docname or Path(docname).name != docname:
        raise ValueError(f"invalid docname: {docname!r}")
    return docname


def rewrite_asset_links_for_top_level(markdown: str, docname: str) -> str:
    """Rewrite internal `assets/` links to `<docname>/assets/` for top-level .md.

    The pipeline works with `assets/...` links (staging dir, integrity
    checks). The deliverable keeps the .md at `<output>/<doc>.md` for
    navigation while assets live in `<output>/<doc>/assets/`, so final
    links must be prefixed. Only inline `(assets/...)` targets are
    rewritten; nothing else changes.
    """
    return markdown.replace("(assets/", f"({docname}/assets/")


def write_output(
    output_dir: Path,
    docname: str,
    markdown: str,
    assets_src: Path,
    report: str,
) -> Deliverable:
    """Write `<doc>.md` (top level) + `<doc>/assets/` + `<doc>/report.md`.

    Deterministic asset names (page-{p:03d}-img-{n:02d}) collide across
    jobs, so sharing a single <output>/assets/ means each job overwrites
    the last and earlier markdowns end up pointing at the wrong figures.
    Each document keeps its own `<output>/<docname>/assets/` +
    `conversion-report.md`, while the markdown itself stays at
    `<output>/<docname>.md` for navigation.
    Final image links are `<docname>/assets/...` (still relative +
    portable).
    """
    _validate_docname(docname)
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_dir = output_dir / docname
    doc_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = doc_dir / "assets"
    # Per-doc dir: safe to reset stale assets from a previous run of the
    # same doc (fewer figures second time must not leave orphans behind).
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    if assets_src.is_dir():
        for item in sorted(assets_src.iterdir()):
            if item.is_file():
                shutil.copy2(item, assets_dir / item.name)
    # GitHub strips one level of backslash-escaping from math before KaTeX;
    # rewrite LaTeX punctuation for that transport (math spans only).
    final_markdown = github_math_safe(markdown)
    final_markdown = rewrite_asset_links_for_top_level(final_markdown, docname)
    markdown_path = output_dir / f"{docname}.md"
    markdown_path.write_text(final_markdown, encoding="utf-8")
    report_path = doc_dir / "conversion-report.md"
    report_path.write_text(report, encoding="utf-8")
    return Deliverable(markdown_path=markdown_path, assets_dir=assets_dir, report_path=report_path)


async def collect_job_pages(session: AsyncSession, job_id: UUID) -> tuple[Job | None, list[Page]]:
    """Job + ordered pages convenience for drivers (6.5/6.8)."""
    from sqlalchemy import select

    job = await repo.get_job(session, job_id)
    if job is None:
        return None, []
    rows = (
        await session.scalars(select(Page).where(Page.job_id == job_id).order_by(Page.page_number))
    ).all()
    return job, list(rows)


async def collect_job_images(session: AsyncSession, job_id: UUID) -> list[Image]:
    """Ordered figure rows for the report's conversion summary."""
    from sqlalchemy import select

    rows = (
        await session.scalars(
            select(Image).where(Image.job_id == job_id).order_by(Image.page_number, Image.id)
        )
    ).all()
    return list(rows)


__all__ = [
    "Deliverable",
    "build_report",
    "collect_job_images",
    "collect_job_pages",
    "rewrite_asset_links_for_top_level",
    "write_output",
]
