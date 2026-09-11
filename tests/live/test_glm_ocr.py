"""live GLM-OCR on 3 sample pages + ocr_output persistence.

LIVE tier (NFR-11): real local model, excluded from default runs
(`-m 'not live'`). Run explicitly at gates: pytest -m live.
"""

import uuid
from pathlib import Path

import pytest

from src.config import load_settings
from src.db import engine as engine_mod
from src.db import repo
from src.db.models import Job, PageStatus
from src.pdf import render_page
from src.pipeline.ocr import GlmOcrEngine

pytestmark = pytest.mark.live


async def test_live_ocr_three_pages_and_persist(tmp_path) -> None:
    settings = load_settings()
    engine = GlmOcrEngine(settings.OLLAMA_LOCAL_URL, model=settings.OCR_MODEL, timeout=600)
    renders = tmp_path / "renders"
    texts: list[str] = []
    for number in (1, 2, 3):
        info = render_page(Path("corpus/bitcoin.pdf"), number, 200, renders / f"p{number}.png")
        assert info.width > 0
        result = await engine.run_page(renders / f"p{number}.png")
        assert len(result.text.strip()) > 100, f"page {number} OCR too short"
        assert result.model == settings.OCR_MODEL
        texts.append(result.text)

    # Page 1 mentions Bitcoin in title or body (ground-truth spot check).
    assert "Bitcoin" in texts[0]

    # Persistence path: ocr_output lands in pages.ocr_output via checkpoint.
    dsn = settings.DATABASE_URL.get_secret_value()
    db_engine = engine_mod.create_engine(dsn)
    try:
        factory = engine_mod.session_factory(db_engine)
        async with factory() as session:
            job = await repo.create_job(
                session,
                filename=f"live-3.1-{uuid.uuid4().hex}.pdf",
                file_sha256="e" * 64,
                page_count=3,
                output_dir=".",
            )
            try:
                await repo.checkpoint_page(
                    session,
                    job.id,
                    1,
                    status=PageStatus.OCR,
                    ocr_output={"prompt": "Text Recognition:", "text": texts[0]},
                )
                back = await repo.find_resume_point(session, job.id)
                assert back == 1  # ocr status is non-terminal: still due
            finally:
                doomed = await session.get(Job, job.id)
                if doomed is not None:
                    await session.delete(doomed)
                await session.commit()
    finally:
        await db_engine.dispose()
