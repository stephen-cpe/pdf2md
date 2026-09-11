"""live Cloud smoke — preflight resolve + 1 transcribe + usage.

LIVE tier: one image-heavy Cloud call. Explicit only: pytest -m live.
"""

from pathlib import Path

import pytest

from src.config import load_settings
from src.pdf import render_page
from src.pipeline.agent import GlmFlashAgent, resolve_cloud_model

pytestmark = pytest.mark.live


async def test_live_cloud_smoke(tmp_path) -> None:
    settings = load_settings()
    key = settings.OLLAMA_API_KEY.get_secret_value()

    resolved = resolve_cloud_model(settings.OLLAMA_CLOUD_URL, key, settings.AGENT_MODEL)
    assert resolved == "glm-5.3-flash"

    render = tmp_path / "smoke.png"
    info = render_page(Path("corpus/bitcoin.pdf"), 1, 150, render)
    agent = GlmFlashAgent(
        settings.OLLAMA_CLOUD_URL,
        key,
        model=resolved,
        thinking_effort=settings.THINKING_EFFORT_TRANSCRIBE,
        timeout=settings.AGENT_TIMEOUT_SECONDS,
    )
    result = await agent.transcribe(
        render,
        "Bitcoin: A Peer-to-Peer Electronic Cash System (canned OCR reference for smoke).",
        info.width,
        info.height,
    )
    assert "<<<MARKDOWN>>>" in result.raw and "<<<END_MARKDOWN>>>" in result.raw
    assert result.prompt_tokens is not None and result.prompt_tokens > 0
    assert result.completion_tokens is not None and result.completion_tokens > 0
    assert result.latency_ms > 0
