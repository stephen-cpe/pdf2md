"""Unit tier: Cloud routing proof, name forms, usage (mocked)."""

from typing import ClassVar

import pytest

from src.pipeline import agent as agent_mod
from src.pipeline.agent import GlmFlashAgent, resolve_cloud_model

CLOUD = "https://ollama.com"
LOCAL = "http://localhost:11434"


class _FakeChatResponse(dict):
    """Dict-shaped stand-in for ollama ChatResponse."""


class _FakeClient:
    seen: ClassVar[dict] = {}

    def __init__(self, host=None, headers=None, **kwargs):
        type(self).seen = {"host": host, "headers": headers or {}, "kwargs": kwargs}

    def chat(self, model=None, messages=None, think=None, **kwargs):
        type(self).seen.update({"model": model, "messages": messages, "think": think})
        return _FakeChatResponse(
            {
                "message": {"content": "<<<MARKDOWN>>>\nhi\n<<<END_MARKDOWN>>>"},
                "prompt_eval_count": 100,
                "eval_count": 20,
            }
        )


@pytest.fixture()
def _fake(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setattr(agent_mod.ollama, "Client", _FakeClient)
    image = tmp_path / "p.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    return image


async def test_cloud_client_never_local(_fake) -> None:
    """The Cloud adapter targets the Cloud host with Bearer — never the daemon."""
    agent = GlmFlashAgent(CLOUD, "secret-key")
    await agent.transcribe(_fake, "ocr ref", 100, 100)
    assert _FakeClient.seen["host"] == CLOUD
    assert _FakeClient.seen["host"] != LOCAL
    assert _FakeClient.seen["headers"]["Authorization"] == "Bearer secret-key"


async def test_loopback_rejected() -> None:
    for bad in (LOCAL, "http://127.0.0.1:11434", "http://localhost:11434"):
        with pytest.raises(ValueError, match="DEC-002"):
            GlmFlashAgent(bad, "k")


async def test_missing_key_rejected() -> None:
    with pytest.raises(ValueError, match="API key"):
        GlmFlashAgent(CLOUD, "")


async def test_thinking_effort_and_usage(_fake) -> None:
    agent = GlmFlashAgent(CLOUD, "k", thinking_effort="high")
    result = await agent.transcribe(_fake, "ocr", 100, 100)
    assert _FakeClient.seen["think"] == "high"
    assert result.thinking_effort == "high"
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 20
    assert result.latency_ms >= 0.0
    assert result.model == "glm-5.3-flash"


async def test_image_dims_in_prompt(_fake) -> None:
    agent = GlmFlashAgent(CLOUD, "k")
    await agent.transcribe(_fake, "ocr", 1700, 2200, page_number=3)
    user_content = _FakeClient.seen["messages"][1]["content"]
    assert "1700x2200" in user_content
    assert _FakeClient.seen["messages"][1]["images"]


class _FakeTags:
    def __init__(self, names: list[str]):
        self._names = names

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, *a, **k):
        return self

    def raise_for_status(self):
        return None

    def json(self):
        return {"models": [{"name": n} for n in self._names]}


def test_resolve_both_name_forms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_mod.httpx, "Client", lambda **k: _FakeTags(["glm-5.3-flash"]))
    assert resolve_cloud_model(CLOUD, "k", "glm-5.3-flash") == "glm-5.3-flash"
    assert resolve_cloud_model(CLOUD, "k", "glm-5.3-flash:cloud") == "glm-5.3-flash"


def test_resolve_unknown_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_mod.httpx, "Client", lambda **k: _FakeTags(["other"]))
    with pytest.raises(ValueError, match="not in Ollama Cloud"):
        resolve_cloud_model(CLOUD, "k", "glm-5.3-flash")
