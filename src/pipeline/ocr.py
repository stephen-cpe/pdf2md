"""OCR stage — OcrEngine interface + local GLM-OCR adapter.

Implements the A.1 prompt contract behind the NFR-10 interface with
telemetry. Async throughout — the sync Ollama client runs in a worker
thread so the event loop never blocks (the look-ahead loop needs this).
Single attempt here; timeout/backoff lives in the resilience wrapper.
Local daemon ONLY: no auth headers, ever.
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import ollama

TEXT_PROMPT = "Text Recognition:"
TABLE_PROMPT = "Table Recognition:"
FORMULA_PROMPT = "Formula Recognition:"


@dataclass(frozen=True)
class OcrResult:
    """Raw OCR output + telemetry for pages.ocr_output / FR-AGT-9."""

    text: str
    prompt: str
    model: str
    latency_ms: float


class OcrEngine(Protocol):
    """Pipeline OCR stage (NFR-10 swappable interface)."""

    async def run_page(self, image: Path, prompt: str = TEXT_PROMPT) -> OcrResult:
        """Recognize one page image; raises on transport failure."""
        ...  # pragma: no cover


class GlmOcrEngine:
    """GLM-OCR via the LOCAL Ollama daemon (FR-OCR-1)."""

    def __init__(self, local_url: str, model: str = "glm-ocr", timeout: int = 120) -> None:
        if not local_url or "ollama.com" in local_url:
            raise ValueError(f"OCR engine requires the local daemon URL, got {local_url!r}")
        self._local_url = local_url
        self._model = model
        self._timeout = timeout

    def _chat(self, image_bytes: bytes, prompt: str) -> str:
        client = ollama.Client(host=self._local_url, timeout=self._timeout)
        response = client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt, "images": [image_bytes]}],
        )
        content = response["message"]["content"]
        return str(content)

    async def run_page(self, image: Path, prompt: str = TEXT_PROMPT) -> OcrResult:
        """Recognize one page; sync client runs off-loop in a thread.

        Hard wall-clock deadline via wait_for: the underlying httpx read
        timeout only fires on idle, but glm-ocr can enter a degenerate
        token-trickle loop (e.g. endless ``` fences on dense code pages
        at 200 DPI) that never idles yet never completes. wait_for caps
        total time and lets resilient pause/retry or pages.py fallback
        to a lower DPI.
        """
        image_bytes = Path(image).read_bytes()
        start = time.perf_counter()
        try:
            text: str = await asyncio.wait_for(
                asyncio.to_thread(self._chat, image_bytes, prompt),
                timeout=float(self._timeout),
            )
        except TimeoutError as exc:
            raise TimeoutError(f"OCR timed out after {self._timeout}s") from exc
        return OcrResult(
            text=text,
            prompt=prompt,
            model=self._model,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )


__all__ = [
    "FORMULA_PROMPT",
    "TABLE_PROMPT",
    "TEXT_PROMPT",
    "GlmOcrEngine",
    "OcrEngine",
    "OcrResult",
]
