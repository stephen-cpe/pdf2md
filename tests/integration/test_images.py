"""Integration: real DB + real files, fake models (5.1/5.2/5.3)."""

import uuid
from pathlib import Path

import pymupdf
import pytest
from sqlalchemy import select

from src.config import load_settings
from src.db import engine as engine_mod
from src.db import repo
from src.db.models import Image, ImageSource, Job
from src.pdf import render_page
from src.pipeline.diagrams import DiagramData, DiagramResult
from src.pipeline.images import FigurePayload, check_assets, resolve_page_figures

pytestmark = pytest.mark.integration


@pytest.fixture()
async def db():
    dsn = load_settings().DATABASE_URL.get_secret_value()
    engine = engine_mod.create_engine(dsn)
    factory = engine_mod.session_factory(engine)
    created: list = []
    yield factory, created
    async with factory() as session:
        for job_id in created:
            job = await session.get(Job, job_id)
            if job is not None:
                await session.delete(job)
        await session.commit()
    await engine.dispose()


def _jpeg_pdf(path: Path) -> bytes:
    render_doc = pymupdf.open()
    render_doc.new_page(width=200, height=100).insert_text((10, 50), "native pixels")
    pix = render_doc[0].get_pixmap(matrix=pymupdf.Matrix(1, 1))
    jpeg = bytes(pix.tobytes("jpg"))
    render_doc.close()
    doc = pymupdf.open()
    doc.new_page(width=400, height=400).insert_image(pymupdf.Rect(10, 10, 210, 110), stream=jpeg)
    doc.save(path)
    doc.close()
    return jpeg


def _plain_pdf(path: Path) -> None:
    doc = pymupdf.open()
    doc.new_page(width=400, height=400).insert_text((72, 72), "no native images here")
    doc.save(path)
    doc.close()


async def _job(factory, created, tmp_path: Path):
    async with factory() as session:
        job = await repo.create_job(
            session,
            filename=f"m5-{uuid.uuid4().hex}.pdf",
            file_sha256="h" * 64,
            page_count=1,
            output_dir=str(tmp_path),
        )
        created.append(job.id)
        return job.id


async def test_native_first_with_caption(db, tmp_path: Path) -> None:
    factory, created = db
    pdf = tmp_path / "doc.pdf"
    jpeg = _jpeg_pdf(pdf)
    job_id = await _job(factory, created, tmp_path)
    assets = tmp_path / "assets"
    payload = [
        FigurePayload(
            index=1, bbox=(0.0, 0.0, 400.0, 400.0), alt="A chart", caption="Fig. 1: A — B"
        )
    ]
    async with factory() as session:
        outcome = await resolve_page_figures(
            session,
            job_id=job_id,
            page_number=1,
            markdown="Text\n\n<!--FIG:1:1:0,0,400,400-->",
            figures=payload,
            pdf_path=pdf,
            renders_dir=tmp_path / "r",
            assets_dir=assets,
        )
        await session.commit()
        updated = outcome.markdown
        assert outcome.resolved == 1 and outcome.orphaned == []
    assert "![A chart](assets/page-001-img-01.jpg)" in updated
    assert "*Fig. 1: A — B*" in updated  # source caption verbatim
    assert (assets / "page-001-img-01.jpg").read_bytes() == jpeg  # original bytes
    async with factory() as session:
        rows = (await session.scalars(select(Image).where(Image.job_id == job_id))).all()
        assert len(rows) == 1
        assert rows[0].source is ImageSource.NATIVE and rows[0].referenced
    assert check_assets(updated, assets).ok


async def test_native_pairing_rejects_icon_inside_figure(db, tmp_path: Path) -> None:
    """Arxiv p4: one badge icon + one flowchart ref must crop, not pair.

    Count/order alignment (1 native, 1 ref) used to paste the icon's bytes
    under the flowchart's caption and drop the real figure. Geometry
    matching rejects the 0.3%-area icon, so the agent box crops from the
    render instead.
    """
    factory, created = db
    pdf = tmp_path / "doc.pdf"
    render_doc = pymupdf.open()
    render_doc.new_page(width=200, height=100).insert_text((10, 50), "native pixels")
    pix = render_doc[0].get_pixmap(matrix=pymupdf.Matrix(1, 1))
    jpeg = bytes(pix.tobytes("jpg"))
    render_doc.close()
    doc = pymupdf.open()
    doc.new_page(width=400, height=400).insert_image(pymupdf.Rect(10, 10, 60, 60), stream=jpeg)
    doc.save(pdf)
    doc.close()
    renders = tmp_path / "r"
    render_page(pdf, 1, 200, renders / "page-001.png")  # 1111x1111px
    job_id = await _job(factory, created, tmp_path)
    assets = tmp_path / "assets"
    payload = [
        FigurePayload(index=1, bbox=(0.0, 0.0, 600.0, 320.0), alt="Big flowchart", caption=None)
    ]
    async with factory() as session:
        outcome = await resolve_page_figures(
            session,
            job_id=job_id,
            page_number=1,
            markdown="<!--FIG:page:1:0,0,600,320-->",
            figures=payload,
            pdf_path=pdf,
            renders_dir=renders,
            assets_dir=assets,
        )
        await session.commit()
        updated = outcome.markdown
    assert outcome.resolved == 1
    assert "![Big flowchart](assets/page-001-img-01.png)" in updated
    assert (assets / "page-001-img-01.png").read_bytes() != jpeg  # render crop, not icon
    async with factory() as session:
        rows = (await session.scalars(select(Image).where(Image.job_id == job_id))).all()
        assert len(rows) == 1 and rows[0].source is ImageSource.CROP
    assert check_assets(updated, assets).ok


async def test_crop_fallback_truncates_alt_and_details(db, tmp_path: Path) -> None:
    factory, created = db
    pdf = tmp_path / "doc.pdf"
    _plain_pdf(pdf)
    renders = tmp_path / "r"
    info = render_page(pdf, 1, 200, renders / "page-001.png")
    job_id = await _job(factory, created, tmp_path)
    assets = tmp_path / "assets"
    long_alt = "Word " * 40  # 200 chars: forces truncation + details block
    payload = [FigurePayload(index=1, bbox=(10.0, 10.0, 110.0, 60.0), alt=long_alt, caption=None)]
    w, h = info.width, info.height
    assert w > 110 and h > 60
    async with factory() as session:
        outcome = await resolve_page_figures(
            session,
            job_id=job_id,
            page_number=1,
            markdown="<!--FIG:page:1:10,10,110,60-->",
            figures=payload,
            pdf_path=pdf,
            renders_dir=renders,
            assets_dir=assets,
            fig_details=True,
        )
        await session.commit()
        updated = outcome.markdown
    link_line = next(line for line in updated.splitlines() if line.startswith("!["))
    alt_shown = link_line[2 : link_line.index("]")]
    assert len(alt_shown) <= 125
    assert "<details>" in updated and "Word Word" in updated
    async with factory() as session:
        rows = (await session.scalars(select(Image).where(Image.job_id == job_id))).all()
        assert rows[0].source is ImageSource.CROP
    assert check_assets(updated, assets).ok


async def test_details_toggle_off(db, tmp_path: Path) -> None:
    factory, created = db
    pdf = tmp_path / "doc.pdf"
    _plain_pdf(pdf)
    renders = tmp_path / "r"
    render_page(pdf, 1, 200, renders / "page-001.png")
    job_id = await _job(factory, created, tmp_path)
    payload = [FigurePayload(index=1, bbox=(10.0, 10.0, 110.0, 60.0), alt="Word " * 40)]
    async with factory() as session:
        outcome = await resolve_page_figures(
            session,
            job_id=job_id,
            page_number=1,
            markdown="<!--FIG:1:1:10,10,110,60-->",
            figures=payload,
            pdf_path=pdf,
            renders_dir=renders,
            assets_dir=tmp_path / "assets",
            fig_details=False,
        )
        await session.commit()
        updated = outcome.markdown
    assert "<details>" not in updated


async def test_clear_job_images(db, tmp_path: Path) -> None:
    factory, created = db
    async with factory() as session:
        job = await repo.create_job(
            session,
            filename=f"m5-clear-{uuid.uuid4().hex}.pdf",
            file_sha256="i" * 64,
            page_count=1,
            output_dir=str(tmp_path),
        )
        created.append(job.id)
        for index in (1, 2):
            await repo.record_image(
                session,
                job_id=job.id,
                page_number=1,
                asset_path=f"assets/page-001-img-0{index}.png",
                source=ImageSource.NATIVE,
            )
        await session.commit()
        assert await repo.clear_job_images(session, job.id) == 2
        await session.commit()
        rows = (await session.scalars(select(Image).where(Image.job_id == job.id))).all()
        assert rows == []


async def test_orphan_payload_appended_never_lost(db, tmp_path: Path) -> None:
    """NFR-1: a FIGURES entry with no markdown token is appended, not dropped."""
    factory, created = db
    pdf = tmp_path / "doc.pdf"
    _plain_pdf(pdf)
    renders = tmp_path / "r"
    render_page(pdf, 1, 200, renders / "page-001.png")
    job_id = await _job(factory, created, tmp_path)
    assets = tmp_path / "assets"
    payload = [
        FigurePayload(index=1, bbox=(10.0, 10.0, 110.0, 60.0), alt="Lost chart", caption="Fig. 9")
    ]
    async with factory() as session:
        outcome = await resolve_page_figures(
            session,
            job_id=job_id,
            page_number=1,
            markdown="Just prose, no token.",
            figures=payload,
            pdf_path=pdf,
            renders_dir=renders,
            assets_dir=assets,
        )
        await session.commit()
    assert outcome.resolved == 1
    assert outcome.orphaned and outcome.orphaned[0]["placed"] is True
    assert "![Lost chart](assets/page-001-img-01.png)" in outcome.markdown
    assert "*Fig. 9*" in outcome.markdown
    assert (assets / "page-001-img-01.png").is_file()
    async with factory() as session:
        rows = (await session.scalars(select(Image).where(Image.job_id == job_id))).all()
        assert len(rows) == 1 and rows[0].source is ImageSource.CROP
    assert check_assets(outcome.markdown, assets).ok


async def test_unresolvable_left_for_qa(db, tmp_path: Path) -> None:
    factory, created = db
    pdf = tmp_path / "doc.pdf"
    _plain_pdf(pdf)
    renders = tmp_path / "r"
    render_page(pdf, 1, 200, renders / "page-001.png")
    job_id = await _job(factory, created, tmp_path)
    assets = tmp_path / "assets"
    payload = [FigurePayload(index=1, bbox=(0.0, 0.0, 99999.0, 99999.0), alt="x")]
    async with factory() as session:
        outcome = await resolve_page_figures(
            session,
            job_id=job_id,
            page_number=1,
            markdown="<!--FIG:1:1:0,0,99999,99999-->",
            figures=payload,
            pdf_path=pdf,
            renders_dir=renders,
            assets_dir=assets,
        )
        await session.commit()
        updated = outcome.markdown
    assert "<!--FIG:1:1:0,0,99999,99999-->" in updated
    report = check_assets(updated, assets)
    assert not report.ok and len(report.leftover_placeholders) == 1


async def test_mermaid_representation_with_image_fallback(db, tmp_path: Path) -> None:
    """A verified Mermaid reinterpretation replaces the token; original image kept."""
    factory, created = db
    pdf = tmp_path / "doc.pdf"
    _plain_pdf(pdf)
    renders = tmp_path / "r"
    render_page(pdf, 1, 200, renders / "page-001.png")
    job_id = await _job(factory, created, tmp_path)
    assets = tmp_path / "assets"
    diagram = DiagramResult(
        convertible=True,
        mermaid="flowchart TD\n  A[Start] --> B[End]",
        diagram_type="flowchart",
        confidence=92,
        description="A simple flow",
    )
    payload = [FigurePayload(index=1, bbox=(10.0, 10.0, 110.0, 60.0), alt="flow", diagram=diagram)]
    async with factory() as session:
        outcome = await resolve_page_figures(
            session,
            job_id=job_id,
            page_number=1,
            markdown="<!--FIG:page:1:10,10,110,60-->",
            figures=payload,
            pdf_path=pdf,
            renders_dir=renders,
            assets_dir=assets,
            keep_image=True,
        )
        await session.commit()
        updated = outcome.markdown
    assert "```mermaid" in updated and "A[Start] --> B[End]" in updated
    assert "Original figure" in updated  # collapsible fallback
    assert "![flow](assets/page-001-img-01.png)" in updated
    assert check_assets(updated, assets).ok
    async with factory() as session:
        rows = (await session.scalars(select(Image).where(Image.job_id == job_id))).all()
        assert len(rows) == 1
        assert rows[0].conversion_status == "mermaid"
        assert rows[0].diagram_type == "flowchart" and rows[0].confidence == 92
        assert rows[0].mermaid and rows[0].mermaid.startswith("flowchart")


async def test_mermaid_keep_image_off(db, tmp_path: Path) -> None:
    factory, created = db
    pdf = tmp_path / "doc.pdf"
    _plain_pdf(pdf)
    renders = tmp_path / "r"
    render_page(pdf, 1, 200, renders / "page-001.png")
    job_id = await _job(factory, created, tmp_path)
    assets = tmp_path / "assets"
    diagram = DiagramResult(
        convertible=True, mermaid="flowchart LR\n  A --> B", diagram_type="flowchart", confidence=90
    )
    payload = [FigurePayload(index=1, bbox=(10.0, 10.0, 110.0, 60.0), alt="flow", diagram=diagram)]
    async with factory() as session:
        outcome = await resolve_page_figures(
            session,
            job_id=job_id,
            page_number=1,
            markdown="<!--FIG:page:1:10,10,110,60-->",
            figures=payload,
            pdf_path=pdf,
            renders_dir=renders,
            assets_dir=assets,
            keep_image=False,
        )
        await session.commit()
        updated = outcome.markdown
    assert "```mermaid" in updated
    assert "Original figure" not in updated
    # No image link and no orphan asset: integrity stays valid.
    assert "assets/" not in updated
    assert check_assets(updated, assets).ok
    async with factory() as session:
        rows = (await session.scalars(select(Image).where(Image.job_id == job_id))).all()
        assert len(rows) == 1 and rows[0].conversion_status == "mermaid"
        assert rows[0].asset_path == ""


async def test_data_chart_table_fallback(db, tmp_path: Path) -> None:
    """A data chart that fails Mermaid conversion falls back to a grounded table."""
    factory, created = db
    pdf = tmp_path / "doc.pdf"
    _plain_pdf(pdf)
    renders = tmp_path / "r"
    render_page(pdf, 1, 200, renders / "page-001.png")
    job_id = await _job(factory, created, tmp_path)
    assets = tmp_path / "assets"
    diagram = DiagramResult(
        convertible=False,
        diagram_type="pie",
        confidence=40,
        description="A pie chart",
        data=DiagramData(columns=("slice", "value"), rows=(("a", "60"), ("b", "40"))),
        reason="verifier rejected Mermaid (fidelity 40)",
    )
    payload = [FigurePayload(index=1, bbox=(10.0, 10.0, 110.0, 60.0), alt="pie", diagram=diagram)]
    async with factory() as session:
        outcome = await resolve_page_figures(
            session,
            job_id=job_id,
            page_number=1,
            markdown="<!--FIG:page:1:10,10,110,60-->",
            figures=payload,
            pdf_path=pdf,
            renders_dir=renders,
            assets_dir=assets,
            fallback="both",
        )
        await session.commit()
        updated = outcome.markdown
    assert "| slice | value |" in updated and "| a | 60 |" in updated
    assert "![pie](assets/page-001-img-01.png)" in updated
    assert "```mermaid" not in updated
    assert check_assets(updated, assets).ok
    async with factory() as session:
        rows = (await session.scalars(select(Image).where(Image.job_id == job_id))).all()
        assert rows[0].conversion_status == "table"
