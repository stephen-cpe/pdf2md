"""Unit tier: adapter logic with a mocked Ollama client."""

from pathlib import Path
from typing import ClassVar

import pytest

from src.pipeline import ocr as ocr_mod
from src.pipeline.ocr import TEXT_PROMPT, GlmOcrEngine


class _FakeClient:
    """Captures construction + chat args; returns canned content."""

    seen: ClassVar[dict] = {}

    def __init__(self, host=None, **kwargs):
        type(self).seen = {"host": host, "kwargs": kwargs}

    def chat(self, model=None, messages=None, **kwargs):
        type(self).seen.update({"model": model, "messages": messages})
        return {"message": {"content": "MOCKED OCR TEXT"}}


@pytest.fixture()
def _fake(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Path:
    monkeypatch.setattr(ocr_mod.ollama, "Client", _FakeClient)
    image = tmp_path / "p.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    return image


async def test_prompt_and_text_passthrough(_fake: Path) -> None:
    engine = GlmOcrEngine("http://localhost:11434")
    result = await engine.run_page(_fake, TEXT_PROMPT)
    assert result.text == "MOCKED OCR TEXT"
    assert result.prompt == TEXT_PROMPT
    assert result.model == "glm-ocr"
    assert result.latency_ms >= 0.0
    assert _FakeClient.seen["messages"][0]["content"] == TEXT_PROMPT


async def test_local_routing_no_cloud(_fake: Path) -> None:
    engine = GlmOcrEngine("http://localhost:11434", model="glm-ocr")
    await engine.run_page(_fake)
    assert _FakeClient.seen["host"] == "http://localhost:11434"
    assert "Authorization" not in _FakeClient.seen["kwargs"]
    assert _FakeClient.seen["model"] == "glm-ocr"


async def test_cloud_url_rejected() -> None:
    with pytest.raises(ValueError, match="local daemon"):
        GlmOcrEngine("https://ollama.com")


async def test_table_prompt_forwarded(_fake: Path) -> None:
    engine = GlmOcrEngine("http://localhost:11434")
    result = await engine.run_page(_fake, ocr_mod.TABLE_PROMPT)
    assert result.prompt == ocr_mod.TABLE_PROMPT
    assert _FakeClient.seen["messages"][0]["content"] == ocr_mod.TABLE_PROMPT
