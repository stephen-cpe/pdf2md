"""Unit tier: Mermaid validation, conversion gating, tiered fallback."""

import pytest

from src.pipeline.agent import AgentResult
from src.pipeline.diagrams import (
    DiagramData,
    DiagramResult,
    choose_representation,
    convert_figure,
    detect_mermaid_type,
    is_data_chart,
    strip_fences,
    validate_mermaid,
)

ALLOWED = frozenset({"flowchart", "sequenceDiagram", "pie", "xychart-beta"})


def _diagram_envelope(
    *,
    convertible: bool = True,
    dtype: str = "flowchart",
    confidence: int = 90,
    mermaid: str = "flowchart TD\n  A[Start] --> B[End]",
    description: str = "a flow",
    data: str = "{}",
) -> str:
    return (
        "<<<DIAGRAM>>>\n"
        f'{{"convertible": {str(convertible).lower()}, "type": "{dtype}", '
        f'"confidence": {confidence}, "description": "{description}"}}\n'
        "<<<END_DIAGRAM>>>\n"
        "<<<MERMAID>>>\n" + mermaid + "\n<<<END_MERMAID>>>\n"
        "<<<DATA>>>\n" + data + "\n<<<END_DATA>>>"
    )


def _verdict(coverage: int, verdict: str = "pass") -> str:
    return (
        "<<<VERDICT>>>\n"
        f'{{"coverage": {coverage}, "misses": [], "structure_issues": [], "verdict": "{verdict}"}}\n'
        "<<<END_VERDICT>>>"
    )


class _Agent:
    def __init__(self, raws: list[str]):
        self.raws = list(raws)
        self.calls = 0

    async def transcribe(self, image, ocr_text, w, h, n=1, ctx="", outline=""):
        self.calls += 1
        raw = self.raws.pop(0) if len(self.raws) > 1 else self.raws[0]
        return AgentResult(raw, "fake", "high", 1.0, 10, 5)


@pytest.fixture()
def crop(tmp_path):
    image = tmp_path / "fig.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    return image


def test_detect_and_strip_fences() -> None:
    assert detect_mermaid_type("graph TD\n A-->B") == "flowchart"
    assert detect_mermaid_type("sequenceDiagram\n A->>B: hi") == "sequenceDiagram"
    assert detect_mermaid_type("stateDiagram-v2\n [*] --> A") == "stateDiagram-v2"
    assert detect_mermaid_type("%% comment\nflowchart LR\n A-->B") == "flowchart"
    assert detect_mermaid_type("not a diagram") is None
    assert strip_fences("```mermaid\nflowchart TD\n A-->B\n```") == "flowchart TD\n A-->B"


def test_validate_mermaid_accepts_and_rejects() -> None:
    assert validate_mermaid("flowchart TD\n A-->B", ALLOWED) is None
    assert validate_mermaid("", ALLOWED) is not None
    assert "unrecognized" in (validate_mermaid("hello world", ALLOWED) or "")
    assert "allowlist" in (validate_mermaid("classDiagram\n A <|-- B", ALLOWED) or "")
    assert "no body" in (validate_mermaid("flowchart TD", ALLOWED) or "")
    assert "leftover" in (validate_mermaid("flowchart TD\n A<!--FIG:x-->", ALLOWED) or "")


def test_is_data_chart() -> None:
    assert is_data_chart("pie") and is_data_chart("xychart-beta")
    assert not is_data_chart("flowchart")


async def test_convert_success_with_verify(crop) -> None:
    converter = _Agent([_diagram_envelope()])
    verifier = _Agent([_verdict(95)])
    result = await convert_figure(
        converter,
        verifier,
        crop,
        "grounding text",
        page_number=1,
        width=100,
        height=50,
        allowed_types=ALLOWED,
        min_confidence=80,
    )
    assert result.convertible and result.diagram_type == "flowchart"
    assert result.confidence == 95  # max(self-reported, verifier)
    assert verifier.calls == 1


async def test_convert_rejected_by_verifier(crop) -> None:
    converter = _Agent([_diagram_envelope()])
    verifier = _Agent([_verdict(40, "retry")])
    result = await convert_figure(
        converter,
        verifier,
        crop,
        "",
        page_number=1,
        width=100,
        height=50,
        allowed_types=ALLOWED,
        min_confidence=80,
    )
    assert not result.convertible and "verifier rejected" in result.reason


async def test_convert_low_confidence(crop) -> None:
    converter = _Agent([_diagram_envelope(confidence=50)])
    result = await convert_figure(
        converter,
        None,
        crop,
        "",
        page_number=1,
        width=100,
        height=50,
        allowed_types=ALLOWED,
        min_confidence=80,
        verify=False,
    )
    assert not result.convertible and "confidence" in result.reason


async def test_convert_not_convertible_preserves_data(crop) -> None:
    converter = _Agent(
        [
            _diagram_envelope(
                convertible=False,
                dtype="pie",
                data='{"columns": ["slice", "value"], "rows": [["a", "1"]]}',
            )
        ]
    )
    result = await convert_figure(
        converter,
        None,
        crop,
        "",
        page_number=1,
        width=100,
        height=50,
        allowed_types=ALLOWED,
        min_confidence=80,
        verify=False,
    )
    assert not result.convertible
    assert result.data is not None and result.data.columns == ("slice", "value")


async def test_convert_malformed_envelope(crop) -> None:
    converter = _Agent(["garbage"])
    result = await convert_figure(
        converter,
        None,
        crop,
        "",
        page_number=1,
        width=100,
        height=50,
        allowed_types=ALLOWED,
        min_confidence=80,
        verify=False,
    )
    assert not result.convertible and "malformed" in result.reason


def test_choose_representation_tiers() -> None:
    ok = DiagramResult(convertible=True, mermaid="flowchart TD\n A-->B", diagram_type="flowchart")
    assert choose_representation(ok).kind == "mermaid"

    chart = DiagramResult(
        convertible=False,
        diagram_type="pie",
        data=DiagramData(columns=("a",), rows=(("1",),)),
        reason="verifier rejected",
    )
    assert choose_representation(chart, fallback="both").kind == "table"
    assert choose_representation(chart, fallback="image").kind == "image"

    photo = DiagramResult(convertible=False, diagram_type="flowchart", reason="photo")
    assert choose_representation(photo, fallback="both").kind == "image"


def test_diagram_data_markdown_table() -> None:
    data = DiagramData(columns=("name", "value"), rows=(("a", "1"), ("b", "2")))
    table = data.to_markdown_table()
    assert table.splitlines()[0] == "| name | value |"
    assert "| a | 1 |" in table and "| b | 2 |" in table
    assert DiagramData(columns=(), rows=()).to_markdown_table() == ""
