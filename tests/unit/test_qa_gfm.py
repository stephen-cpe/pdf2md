"""Unit tier: patch anchoring, QA envelope, lint classification."""

import pytest

from src.pipeline.envelope import EnvelopeError, Patch, parse_qa
from src.pipeline.gfm import classify, lint_markdown, normalize_markdown
from src.pipeline.qa import QaOutcome, apply_patches

DOC = "# Intro\n\nHello world.\n\n## Detail\n\nSome text here.\n"


def test_apply_patch_exact_once() -> None:
    outcome = apply_patches(DOC, [Patch("Detail", "Some text", "Other text")])
    assert "Other text here." in outcome.markdown
    assert len(outcome.applied) == 1 and not outcome.rejected


def test_reject_missing_section() -> None:
    outcome = apply_patches(DOC, [Patch("Nope", "x", "y")])
    assert not outcome.applied and outcome.rejected[0].reason == "no such section"


def test_reject_ambiguous_find() -> None:
    doc = "# S\n\nsame same\n"
    outcome = apply_patches(doc, [Patch("S", "same", "other")])
    assert not outcome.applied and "exactly 1" in outcome.rejected[0].reason
    assert outcome.markdown == doc


def test_reject_cross_section_find() -> None:
    # "Hello world." lives in Intro, not Detail: scoped regions protect it.
    outcome = apply_patches(DOC, [Patch("Detail", "Hello world.", "Hi.")])
    assert not outcome.applied


def test_patch_log_reversible() -> None:
    patch = Patch("Detail", "Some text", "Other text")
    outcome = apply_patches(DOC, [patch])
    log = outcome.applied[0]
    reversed_md = outcome.markdown.replace(log.replace, log.find, 1)
    assert "Some text here." in reversed_md


def test_parse_qa_valid_and_bad() -> None:
    raw = (
        "<<<PATCHES>>>\n"
        '[{"section": "Detail", "find": "a", "replace": "b"}]\n'
        "<<<END_PATCHES>>>\n<<<SUMMARY>>>\ndone\n<<<END_SUMMARY>>>"
    )
    parsed = parse_qa(raw)
    assert len(parsed.patches) == 1 and parsed.summary == "done"
    with pytest.raises(EnvelopeError):
        parse_qa(raw.replace('"find": "a"', '"find": ""'))


def test_qa_runner_malformed_means_no_patches() -> None:
    import asyncio

    from src.pipeline.agent import AgentResult
    from src.pipeline.qa import run_qa_pass

    class _BadAgent:
        async def transcribe(self, *args, **kwargs) -> AgentResult:
            return AgentResult("not an envelope", "m", "high", 1.0, 1, 1)

    async def _main() -> QaOutcome:
        return await run_qa_pass(_BadAgent(), DOC, "")

    outcome = asyncio.run(_main())
    assert outcome.markdown == DOC and not outcome.applied
    assert "malformed" in outcome.summary


def test_normalize_idempotent_and_safe() -> None:
    messy = "# T\n\nSome   spaced    text.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n$E=mc^2$\n\n*em* **strong**\n"
    once = normalize_markdown(messy)
    assert normalize_markdown(once) == once
    # Semantics survive padding-style normalization; syntax is GFM-clean.
    compact = once.replace(" ", "")
    for token in ("$E=mc^2$", "*em*", "**strong**", "|1|2|"):
        assert token in compact


def test_lint_clean_and_classify() -> None:
    assert lint_markdown("# T\n\nClean paragraph.\n") == []
    verdict = classify(False, ["missing asset"], ["rule hit"])
    assert verdict.hard_fail and verdict.reasons == ["missing asset"]
    assert classify(True, [], ["rule hit"]).hard_fail is False
