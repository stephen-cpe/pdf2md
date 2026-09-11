"""Log lines carry job/stage/page context; timers measured."""

import json
import logging

import pytest

from src.logging import (
    StageTimer,
    clear_context,
    configure_logging,
    get_logger,
    set_context,
)


@pytest.fixture(autouse=True)
def _fresh_logging(caplog: pytest.LogCaptureFixture):
    """Fresh pdf2md handlers per test + capture via caplog."""
    root = logging.getLogger("pdf2md")
    old_handlers = root.handlers[:]
    for handler in old_handlers:
        root.removeHandler(handler)
    clear_context()
    root.addHandler(caplog.handler)
    yield caplog
    root.removeHandler(caplog.handler)
    for handler in old_handlers:
        root.addHandler(handler)
    clear_context()


def _our_formatter(caplog: pytest.LogCaptureFixture) -> logging.Formatter:
    root = logging.getLogger("pdf2md")
    formatters = [
        h.formatter
        for h in root.handlers
        if getattr(h, "_pdf2md_handler", False) and h.formatter is not None
    ]
    assert len(formatters) == 1
    return formatters[0]


def _records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name.startswith("pdf2md.")]


def test_text_lines_carry_context(caplog: pytest.LogCaptureFixture) -> None:
    configure_logging(json_format=False)
    logger = get_logger("test")
    set_context(job_id="job-1", stage="transcribing", page=42)
    with caplog.at_level(logging.INFO, logger="pdf2md"):
        logger.info("page done")
    assert len(_records(caplog)) == 1
    formatted = _our_formatter(caplog).format(_records(caplog)[0])
    assert "job-1" in formatted and "transcribing" in formatted and "42" in formatted


def test_clear_context_unbinds(caplog: pytest.LogCaptureFixture) -> None:
    configure_logging(json_format=False)
    set_context(job_id="job-1", stage="ocr", page=1)
    clear_context()
    with caplog.at_level(logging.INFO, logger="pdf2md"):
        get_logger("test").info("hello")
    formatted = _our_formatter(caplog).format(_records(caplog)[0])
    assert "job-1" not in formatted


def test_json_lines_parse_with_context(caplog: pytest.LogCaptureFixture) -> None:
    configure_logging(json_format=True)
    set_context(job_id="job-9", stage="qa", page=7)
    with caplog.at_level(logging.INFO, logger="pdf2md"):
        get_logger("test").info("qa pass")
    formatted = _our_formatter(caplog).format(_records(caplog)[0])
    payload = json.loads(formatted)
    assert payload["job_id"] == "job-9"
    assert payload["stage"] == "qa"
    assert payload["page"] == 7
    assert payload["message"] == "qa pass"


def test_stage_timer_measures(caplog: pytest.LogCaptureFixture) -> None:
    configure_logging()
    with (
        caplog.at_level(logging.INFO, logger="pdf2md"),
        StageTimer("transcribe", get_logger("test")) as timer,
    ):
        pass
    assert timer.elapsed_ms >= 0.0
    messages = [r.getMessage() for r in _records(caplog)]
    assert "stage started" in messages and "stage finished" in messages


def test_stage_timer_failure_logs_and_reraises(caplog: pytest.LogCaptureFixture) -> None:
    configure_logging()
    with (
        caplog.at_level(logging.INFO, logger="pdf2md"),
        pytest.raises(ValueError, match="boom"),
        StageTimer("ocr", get_logger("test")),
    ):
        raise ValueError("boom")
    messages = [r.getMessage() for r in _records(caplog)]
    assert "stage failed" in messages
