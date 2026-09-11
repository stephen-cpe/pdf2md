"""Unit tier: report fields, writer layout, section splitting."""

from src.db.models import Job, JobStatus, Page, PageStatus
from src.pipeline.report import build_report, split_sections, write_output


def _job() -> Job:
    return Job(
        filename="doc.pdf",
        file_sha256="a" * 64,
        page_count=2,
        output_dir=".",
        status=JobStatus.TRANSCRIBING,
        options={
            "agent_model": "glm-5.3-flash",
            "ocr_model": "glm-ocr",
            "embed_model": "qwen3-embedding:0.6b",
            "render_dpi": 200,
        },
    )


def _page(number: int, review: bool = False) -> Page:
    return Page(
        job_id="00000000-0000-0000-0000-000000000000",
        page_number=number,
        status=PageStatus.NEEDS_REVIEW if review else PageStatus.VERIFIED,
        render_dpi=200,
        markdown=f"# P{number}",
        coverage_score=90 + number,
        retries=1 if review else 0,
        needs_review=review,
        omissions={"notes": "typo fixed: Theorum->Theorem"} if review else {},
        token_usage={"prompt": 10, "completion": 5},
        timings={
            "image_width": 100,
            "image_height": 100,
            "image_bytes": 1000,
            "model": "glm-5.3-flash",
            "thinking_effort": "low",
            "transcribe_ms": 1.0,
            "verify_ms": 1.0,
            "ocr_ms": 1.0,
        },
        source_page_hash="s",
        render_hash="r",
        ocr_hash="o",
        markdown_hash="m",
    )


def test_report_has_every_ac7_field() -> None:
    job = _job()
    report = build_report(
        job,
        [_page(1), _page(2, review=True)],
        pipeline_version="abc123",
        prompt_versions={"transcription": "transcription-v1"},
        furniture_removed=[],
        qa_applied=1,
        qa_rejected=0,
        lint_warnings=["MD001:1:heading"],
    )
    for token in (
        "job_id",
        "wall-clock",
        "tokens: prompt=20 completion=10",
        "pipeline_version: abc123",
        "transcription-v1",
        "glm-5.3-flash",
        "| 1 | verified | 91 |",
        "| 2 | needs_review | 92 |",
        "[2]",
        "Theorum->Theorem",
        "transcribe_ms",
        "s/r/o/m",
        "MD001",
    ):
        assert token in report, token


def test_write_output_layout(tmp_path) -> None:
    assets_src = tmp_path / "src-assets"
    assets_src.mkdir()
    (assets_src / "page-001-img-01.png").write_bytes(b"img")
    deliverable = write_output(
        tmp_path / "out",
        "doc",
        "# Doc\n![a](assets/page-001-img-01.png)\n",
        assets_src,
        "# Report\n",
    )
    # .md top-level for navigation; assets/report in per-doc folder.
    assert deliverable.markdown_path == tmp_path / "out" / "doc.md"
    assert deliverable.assets_dir == tmp_path / "out" / "doc" / "assets"
    assert deliverable.report_path == tmp_path / "out" / "doc" / "conversion-report.md"
    assert deliverable.markdown_path.read_text(encoding="utf-8") == (
        "# Doc\n![a](doc/assets/page-001-img-01.png)\n"
    )
    assert (deliverable.assets_dir / "page-001-img-01.png").read_bytes() == b"img"
    assert deliverable.report_path.name == "conversion-report.md"


def test_write_output_rewrites_asset_links_top_level(tmp_path) -> None:
    from src.pipeline.report import rewrite_asset_links_for_top_level
    from src.pipeline.report import write_output as _write

    assert rewrite_asset_links_for_top_level("![a](assets/x.png)", "bitcoin") == (
        "![a](bitcoin/assets/x.png)"
    )
    assert rewrite_asset_links_for_top_level("no assets here", "bitcoin") == "no assets here"
    src = tmp_path / "src"
    src.mkdir()
    (src / "page-001-img-01.png").write_bytes(b"img")
    deliverable = _write(
        tmp_path / "out",
        "bitcoin",
        "![a](assets/page-001-img-01.png)\n",
        src,
        "# r\n",
    )
    assert "(bitcoin/assets/page-001-img-01.png)" in deliverable.markdown_path.read_text(
        encoding="utf-8"
    )


def test_write_output_isolates_docs_sharing_output_dir(tmp_path) -> None:
    """Two docs to the same output dir must not overwrite each other's assets."""
    from src.pipeline.report import write_output as _write

    for doc, payload in (("bitcoin", b"btc"), ("ocp", b"ocp")):
        src = tmp_path / f"src-{doc}"
        src.mkdir()
        (src / "page-002-img-01.png").write_bytes(payload)
        _write(
            tmp_path / "out",
            doc,
            f"# {doc}\n![a](assets/page-002-img-01.png)\n",
            src,
            f"# {doc} report\n",
        )
    assert (tmp_path / "out" / "bitcoin" / "assets" / "page-002-img-01.png").read_bytes() == b"btc"
    assert (tmp_path / "out" / "ocp" / "assets" / "page-002-img-01.png").read_bytes() == b"ocp"
    assert "(bitcoin/assets/page-002-img-01.png)" in (tmp_path / "out" / "bitcoin.md").read_text(
        encoding="utf-8"
    )
    assert "(ocp/assets/page-002-img-01.png)" in (tmp_path / "out" / "ocp.md").read_text(
        encoding="utf-8"
    )
    assert (tmp_path / "out" / "bitcoin" / "conversion-report.md").read_text(
        encoding="utf-8"
    ) == "# bitcoin report\n"
    assert (tmp_path / "out" / "ocp" / "conversion-report.md").read_text(
        encoding="utf-8"
    ) == "# ocp report\n"


def test_write_output_cleans_stale_assets_same_doc(tmp_path) -> None:
    from src.pipeline.report import write_output as _write

    src1 = tmp_path / "src1"
    src1.mkdir()
    (src1 / "page-001-img-01.png").write_bytes(b"a")
    (src1 / "page-002-img-01.png").write_bytes(b"b")
    _write(tmp_path / "out", "doc", "# v1\n", src1, "# r1\n")
    src2 = tmp_path / "src2"
    src2.mkdir()
    (src2 / "page-001-img-01.png").write_bytes(b"a2")
    second = _write(tmp_path / "out", "doc", "# v2\n", src2, "# r2\n")
    assert (second.assets_dir / "page-001-img-01.png").read_bytes() == b"a2"
    assert not (second.assets_dir / "page-002-img-01.png").exists()


def test_write_output_rejects_unsafe_docname(tmp_path) -> None:
    import pytest as _pytest

    from src.pipeline.report import write_output as _write

    src = tmp_path / "src"
    src.mkdir()
    for bad in ("", ".", "..", "a/b", "a\\b", "../evil"):
        with _pytest.raises(ValueError):
            _write(tmp_path / "out", bad, "# x\n", src, "# r\n")


def test_split_sections() -> None:
    sections = split_sections("# A\n\ntext a\n\n## B\n\ntext b\n")
    assert [heading for heading, _ in sections] == ["A", "B"]
    assert "text b" in sections[1][1]
    assert split_sections("") == []


def test_persist_document_embeddings_uses_tmp_chroma(tmp_path) -> None:
    from src.chroma import ChromaStore
    from src.pipeline.report import persist_document_embeddings

    store = ChromaStore(
        str(tmp_path / "chroma"),
        "fake",
        embed_fn=lambda texts: [[1.0, 0.0] if "text a" in t else [0.0, 1.0] for t in texts],
    )
    count = persist_document_embeddings(store, "doc", "# A\n\ntext a\n\n# B\n\ntext b\n")
    # one child chunk per section (heading included, small sections merge)
    assert count == 2
    result = store.query("documents", "text a", n_results=1)
    assert result["ids"][0][0].startswith("doc:")
    metas = result.get("metadatas", [[]])[0]
    assert metas and "A" in metas[0]["heading"] and metas[0]["parent_id"].startswith("doc:")
