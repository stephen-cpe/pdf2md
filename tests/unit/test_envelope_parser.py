"""Fuzz battery: truncation, imbalance, bad payloads, lookalikes."""

import pytest

from src.pipeline.envelope import (
    EnvelopeError,
    corrective_prompt,
    parse_diagram,
    parse_transcription,
    parse_verdict,
)

VALID = """<<<MARKDOWN>>>
# Title

See <<<MARKDOWN>>> docs for details.
<<<END_MARKDOWN>>>
<<<FIGURES>>>
[{"index": 1, "bbox": [2, 2, 392, 359], "alt": "A diagram", "caption": null}]
<<<END_FIGURES>>>
<<<FURNITURE>>>
{"header": "", "footer": "", "page-number": "1"}
<<<END_FURNITURE>>>
<<<NOTES>>>
- nothing omitted
<<<END_NOTES>>>"""

VALID_VERDICT = """<<<VERDICT>>>
{"coverage": 98, "misses": [], "structure_issues": [], "verdict": "pass"}
<<<END_VERDICT>>>"""


def test_valid_transcription() -> None:
    env = parse_transcription(VALID)
    assert env.markdown.startswith("# Title")
    # Mid-line lookalike preserved as content, not treated as a marker.
    assert "<<<MARKDOWN>>> docs" in env.markdown
    assert len(env.figures) == 1
    figure = env.figures[0]
    assert (figure.index, figure.bbox) == (1, (2.0, 2.0, 392.0, 359.0))
    assert figure.caption is None
    assert env.furniture.page_number == "1"
    assert "nothing omitted" in env.notes


def test_valid_verdict() -> None:
    verdict = parse_verdict(VALID_VERDICT)
    assert (verdict.coverage, verdict.verdict) == (98, "pass")


def test_empty_is_error() -> None:
    with pytest.raises(EnvelopeError, match="missing sections"):
        parse_transcription("")


def test_truncated_is_error_with_raw() -> None:
    raw = "<<<MARKDOWN>>>\n# Title\n"
    with pytest.raises(EnvelopeError, match="truncated") as info:
        parse_transcription(raw)
    assert info.value.raw == raw  # raw preserved verbatim, never discarded


def test_missing_section() -> None:
    raw = VALID.split("<<<FIGURES>>>")[0]
    with pytest.raises(EnvelopeError, match="missing sections"):
        parse_transcription(raw)


def test_stray_end_marker() -> None:
    with pytest.raises(EnvelopeError, match="unexpected"):
        parse_transcription("<<<END_MARKDOWN>>>\n" + VALID)


def test_duplicate_section() -> None:
    raw = VALID.replace("<<<END_MARKDOWN>>>", "<<<END_MARKDOWN>>>\n<<<MARKDOWN>>>\nagain")
    with pytest.raises(EnvelopeError, match="duplicate"):
        parse_transcription(raw)


def test_content_outside_section() -> None:
    with pytest.raises(EnvelopeError, match="outside any section"):
        parse_transcription("hello leader\n" + VALID)


def test_indented_marker_is_not_a_marker() -> None:
    # Strict column-0: an indented open marker fails loudly (retry path),
    # it must never silently re-anchor the parse.
    raw = VALID.replace("<<<MARKDOWN>>>", "  <<<MARKDOWN>>>", 1)
    with pytest.raises(EnvelopeError):
        parse_transcription(raw)


def test_invalid_figures_json() -> None:
    raw = VALID.replace('"caption": null}]', '"caption": null"x}]')
    with pytest.raises(EnvelopeError, match="FIGURES JSON"):
        parse_transcription(raw)


@pytest.mark.parametrize(
    "figures",
    [
        '{"not": "an array"}',
        '["just a string"]',
        '[{"index": 1, "alt": "no bbox"}]',
        '[{"index": 1, "bbox": [0, 0, 1], "alt": "short"}]',
        '[{"index": 1, "bbox": [0, 0, 1, 1], "alt": 42}]',
        '[{"index": 1, "bbox": [0, 0, 1, 1], "alt": "x", "caption": 7}]',
    ],
)
def test_malformed_figures_payload(figures: str) -> None:
    raw = VALID.split("<<<FIGURES>>>\n")[0] + "<<<FIGURES>>>\n" + figures + "\n<<<END_FIGURES>>>\n"
    raw += "<<<FURNITURE>>>\n{}\n<<<END_FURNITURE>>>\n<<<NOTES>>>\n\n<<<END_NOTES>>>"
    with pytest.raises(EnvelopeError):
        parse_transcription(raw)


def test_furniture_must_be_object() -> None:
    raw = VALID.replace('{"header": "", "footer": "", "page-number": "1"}', "[]")
    with pytest.raises(EnvelopeError, match="FURNITURE.*object"):
        parse_transcription(raw)


def test_empty_figures_and_furniture_default() -> None:
    raw = (
        "<<<MARKDOWN>>>\nbody\n<<<END_MARKDOWN>>>\n"
        "<<<FIGURES>>>\n\n<<<END_FIGURES>>>\n"
        "<<<FURNITURE>>>\n\n<<<END_FURNITURE>>>\n"
        "<<<NOTES>>>\n\n<<<END_NOTES>>>"
    )
    env = parse_transcription(raw)
    assert env.figures == () and env.furniture.page_number == ""


@pytest.mark.parametrize(
    ("payload", "pattern"),
    [
        ('{"coverage": 101, "misses": [], "structure_issues": [], "verdict": "pass"}', "coverage"),
        ('{"coverage": true, "misses": [], "structure_issues": [], "verdict": "pass"}', "coverage"),
        (
            '{"coverage": 50, "misses": "none", "structure_issues": [], "verdict": "retry"}',
            "misses",
        ),
        ('{"coverage": 50, "misses": [], "structure_issues": [], "verdict": "maybe"}', "verdict"),
        ('{"coverage": 50, "misses": [], "structure_issues": []}', "verdict"),
        ("[1, 2]", "object"),
        ("{oops", "JSON"),
    ],
)
def test_malformed_verdicts(payload: str, pattern: str) -> None:
    with pytest.raises(EnvelopeError, match=pattern):
        parse_verdict(f"<<<VERDICT>>>\n{payload}\n<<<END_VERDICT>>>")


def test_corrective_prompt_names_defect() -> None:
    try:
        parse_transcription("<<<MARKDOWN>>>\nbody\n")
    except EnvelopeError as exc:
        prompt = corrective_prompt(exc)
        assert "truncated" in prompt and "<<<MARKDOWN>>>" in prompt and "column 0" in prompt
    else:  # pragma: no cover
        raise AssertionError("expected EnvelopeError")


VALID_DIAGRAM = """<<<DIAGRAM>>>
{"convertible": true, "type": "flowchart", "confidence": 90, "description": "a flow"}
<<<END_DIAGRAM>>>
<<<MERMAID>>>
flowchart TD
  A[Start] --> B[End]
<<<END_MERMAID>>>
<<<DATA>>>
{"columns": ["x", "y"], "rows": [["a", "1"]]}
<<<END_DATA>>>"""


def test_valid_diagram() -> None:
    parsed = parse_diagram(VALID_DIAGRAM)
    assert parsed.convertible and parsed.diagram_type == "flowchart"
    assert parsed.confidence == 90
    assert parsed.mermaid.startswith("flowchart TD")
    assert parsed.data is not None and parsed.data.columns == ("x", "y")


def test_diagram_not_convertible_with_empty_data() -> None:
    raw = (
        "<<<DIAGRAM>>>\n"
        '{"convertible": false, "type": "", "confidence": 0, "description": "photo"}\n'
        "<<<END_DIAGRAM>>>\n<<<MERMAID>>>\n\n<<<END_MERMAID>>>\n<<<DATA>>>\n\n<<<END_DATA>>>"
    )
    parsed = parse_diagram(raw)
    assert not parsed.convertible and parsed.data is None and parsed.description == "photo"


@pytest.mark.parametrize(
    ("payload", "pattern"),
    [
        ('{"convertible": "yes", "type": "", "confidence": 0}', "convertible"),
        ('{"convertible": true, "type": 7, "confidence": 0}', "type"),
        ('{"convertible": true, "type": "", "confidence": 101}', "confidence"),
        ('{"convertible": true, "type": "", "confidence": true}', "confidence"),
        ("[1, 2]", "object"),
    ],
)
def test_malformed_diagram_payloads(payload: str, pattern: str) -> None:
    raw = (
        f"<<<DIAGRAM>>>\n{payload}\n<<<END_DIAGRAM>>>\n"
        "<<<MERMAID>>>\nflowchart TD\n A-->B\n<<<END_MERMAID>>>\n"
        "<<<DATA>>>\n{}\n<<<END_DATA>>>"
    )
    with pytest.raises(EnvelopeError, match=pattern):
        parse_diagram(raw)


def test_malformed_diagram_data() -> None:
    raw = VALID_DIAGRAM.replace('{"columns": ["x", "y"], "rows": [["a", "1"]]}', '{"columns": "x"}')
    with pytest.raises(EnvelopeError, match="DATA.columns"):
        parse_diagram(raw)
