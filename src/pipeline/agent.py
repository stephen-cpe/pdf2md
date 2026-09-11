"""Transcription stage — interface + Cloud GLM-5.3-Flash adapter.

Dedicated Cloud client: OLLAMA_CLOUD_URL + Bearer key,
never the local daemon — the constructor rejects loopback hosts outright.
Both AGENT_MODEL forms (§5.3) resolve via preflight /api/tags. Thinking
effort maps to the client's think level; token usage is captured per call
(FR-AGT-9). Single attempt; envelope parsing and retries live in the
envelope and resilience modules.
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlparse

import httpx
import ollama

from src.pipeline.prompts import TRANSCRIPTION_TEMPLATE as TRANSCRIPTION_SYSTEM


@dataclass(frozen=True)
class AgentResult:
    """Raw envelope + per-call telemetry (FR-AGT-9/10 foundation)."""

    raw: str
    model: str
    thinking_effort: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None


class TranscriptionAgent(Protocol):
    """Pipeline transcription stage (NFR-10 swappable interface)."""

    async def transcribe(
        self,
        image: Path | None,
        ocr_text: str,
        image_width: int,
        image_height: int,
        page_number: int = 1,
        rolling_context: str = "",
        outline: str = "",
    ) -> AgentResult:
        """Transcribe one page; raises on transport failure."""
        ...  # pragma: no cover


def _is_loopback(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1") or host.startswith("127.")


def resolve_cloud_model(cloud_url: str, api_key: str, model: str, timeout: float = 15.0) -> str:
    """Accept both name forms (§5.3); resolve via preflight /api/tags."""
    with httpx.Client(timeout=timeout) as client:
        response = client.get(
            f"{cloud_url}/api/tags", headers={"Authorization": f"Bearer {api_key}"}
        )
        response.raise_for_status()
        names = [str(m.get("name", "")) for m in response.json().get("models", [])]
    if model in names:
        return model
    bare = model.removesuffix(":cloud")
    if bare in names:
        return bare
    raise ValueError(f"model {model!r} not in Ollama Cloud model list")


class GlmFlashAgent:
    """GLM-5.3-Flash via Ollama Cloud (FR-AGT-1)."""

    def __init__(
        self,
        cloud_url: str,
        api_key: str,
        model: str = "glm-5.3-flash",
        thinking_effort: Literal["low", "medium", "high"] = "low",
        timeout: int = 300,
        system_prompt: str = TRANSCRIPTION_SYSTEM,
    ) -> None:
        if _is_loopback(cloud_url):
            raise ValueError(f"Cloud agent must not target this host: {cloud_url!r} (DEC-002)")
        if not api_key:
            raise ValueError("Cloud agent requires an API key (DEC-001)")
        self._cloud_url = cloud_url
        self._api_key = api_key
        self._model = model
        self._thinking_effort = thinking_effort
        self._timeout = timeout
        self._system_prompt = system_prompt

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _chat(
        self, image_bytes: bytes | None, user_content: str
    ) -> tuple[str, int | None, int | None]:
        client = ollama.Client(host=self._cloud_url, headers=self._headers(), timeout=self._timeout)
        user_message: dict[str, object] = {"role": "user", "content": user_content}
        if image_bytes is not None:
            user_message["images"] = [image_bytes]
        response = client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                user_message,
            ],
            think=self._thinking_effort,
        )
        raw = str(response["message"]["content"])
        prompt_tokens = response.get("prompt_eval_count")
        completion_tokens = response.get("eval_count")
        return (
            raw,
            int(prompt_tokens) if prompt_tokens is not None else None,
            int(completion_tokens) if completion_tokens is not None else None,
        )

    async def transcribe(
        self,
        image: Path | None,
        ocr_text: str,
        image_width: int,
        image_height: int,
        page_number: int = 1,
        rolling_context: str = "",
        outline: str = "",
    ) -> AgentResult:
        """One multimodal Cloud call; sync client runs off-loop in a thread.

        Hard wall-clock deadline via wait_for mirrors the OCR fix:
        Cloud model can also trickle tokens indefinitely without hitting
        the httpx idle timeout. Capping total time lets resilience
        handle it as a retry/pause instead of hanging the pipeline forever.
        """
        context = rolling_context or "none (first page of slice)"
        outline_text = outline or "none"
        user_content = (
            f"Page image dimensions: {image_width}x{image_height} pixels "
            f"(bbox values MUST be absolute pixels in this space).\n"
            f"Page number: {page_number}.\n"
            f"Rolling context:\n{context}\n"
            f"Document outline so far: {outline_text}\n"
            f"OCR reference (trust for characters, not for structure):\n{ocr_text}"
        )
        image_bytes = Path(image).read_bytes() if image is not None else None
        start = time.perf_counter()
        try:
            raw, prompt_tokens, completion_tokens = await asyncio.wait_for(
                asyncio.to_thread(self._chat, image_bytes, user_content),
                timeout=float(self._timeout),
            )
        except TimeoutError as exc:
            raise TimeoutError(f"agent timed out after {self._timeout}s") from exc
        return AgentResult(
            raw=raw,
            model=self._model,
            thinking_effort=self._thinking_effort,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


__all__ = [
    "TRANSCRIPTION_SYSTEM",
    "AgentResult",
    "GlmFlashAgent",
    "TranscriptionAgent",
    "resolve_cloud_model",
]
