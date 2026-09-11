"""Structured logging with job/stage/page context + stage timers.

Usage:
    configure_logging(json_format=False)
    logger = get_logger(__name__)
    set_context(job_id="...", stage="transcribing", page=42)
    logger.info("page done")
    with StageTimer("transcribe", logger):
        ...  # logs start/end with duration_ms

Context rides on contextvars, so concurrent jobs never leak fields into
each other. NFR-9: JSON option, per-stage timings (elapsed_ms is also
exposed for telemetry capture).
"""

import json
import logging
import sys
import time
from contextvars import ContextVar
from types import TracebackType
from typing import Any, Self

_job_id: ContextVar[str | None] = ContextVar("pdf2md_job_id", default=None)
_stage: ContextVar[str | None] = ContextVar("pdf2md_stage", default=None)
_page: ContextVar[int | None] = ContextVar("pdf2md_page", default=None)

_ROOT_NAME = "pdf2md"


class _ContextFilter(logging.Filter):
    """Inject current contextvars into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = _job_id.get()
        record.stage = _stage.get()
        record.page = _page.get()
        return True


class _JsonFormatter(logging.Formatter):
    """One JSON object per line: ts/level/logger/message + context fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "job_id": getattr(record, "job_id", None),
            "stage": getattr(record, "stage", None),
            "page": getattr(record, "page", None),
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _text_format() -> str:
    return (
        "%(asctime)s %(levelname)s %(name)s "
        "job=%(job_id)s stage=%(stage)s page=%(page)s %(message)s"
    )


def configure_logging(level: str = "INFO", json_format: bool = False) -> logging.Logger:
    """Configure the pdf2md root logger once (idempotent). Returns it."""
    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(level)
    root.propagate = False
    if not any(getattr(h, "_pdf2md_handler", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter() if json_format else logging.Formatter(_text_format()))
        handler.addFilter(_ContextFilter())
        handler.__dict__["_pdf2md_handler"] = True
        root.addHandler(handler)
    return root


def get_logger(name: str) -> logging.Logger:
    """Child of the pdf2md root — inherits handlers, context filter, format."""
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


def set_context(
    job_id: str | None = None, stage: str | None = None, page: int | None = None
) -> None:
    """Bind context fields; None arguments leave the current value alone."""
    if job_id is not None:
        _job_id.set(job_id)
    if stage is not None:
        _stage.set(stage)
    if page is not None:
        _page.set(page)


def clear_context() -> None:
    """Unbind all context fields (job boundary / tests)."""
    _job_id.set(None)
    _stage.set(None)
    _page.set(None)


class StageTimer:
    """Time one pipeline stage; logs start/end with duration_ms.

    The exception path logs the error and re-raises — timers never swallow.
    """

    def __init__(self, stage: str, logger: logging.Logger | None = None) -> None:
        self._stage = stage
        self._logger = logger if logger is not None else get_logger("timer")
        self._start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> Self:
        self._start = time.perf_counter()
        self._logger.info("stage started", extra={"timer_stage": self._stage})
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        if exc_type is None:
            self._logger.info(
                "stage finished",
                extra={"timer_stage": self._stage, "duration_ms": round(self.elapsed_ms, 1)},
            )
        else:
            self._logger.exception(
                "stage failed",
                extra={"timer_stage": self._stage, "duration_ms": round(self.elapsed_ms, 1)},
            )


__all__ = [
    "StageTimer",
    "clear_context",
    "configure_logging",
    "get_logger",
    "set_context",
]
