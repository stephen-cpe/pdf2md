"""Deterministic coverage floor — token recall + gate semantics."""

from src.pipeline.coverage import floor_failure, recall, tokenize


def test_tokenize_strips_markdown_noise() -> None:
    tokens = tokenize(
        "# Heading **bold** `code`\n"
        "| a | b |\n"
        "| --- | --- |\n"
        "| cell1 | cell2 |\n"
        "```python\nprint('hi')\n```\n"
        "[label](http://x) ![alt](assets/img.png)\n"
    )
    # heading text + table cells + code + link label survive; markers don't
    assert "heading" in tokens and "bold" in tokens and "code" in tokens
    assert "cell1" in tokens and "cell2" in tokens
    assert "print" in tokens and "hi" in tokens
    assert "label" in tokens and "alt" in tokens
    assert "#" not in tokens and "|" not in tokens and "```" not in tokens
    assert "http" not in tokens  # link target is dropped, label kept
    assert "assets" not in tokens  # image path dropped, alt kept


def test_tokenize_lowercases_and_splits_on_nonalnum() -> None:
    assert tokenize("Hello, World! 123x A-B") == ["hello", "world", "123x", "a", "b"]


def test_perfect_recall() -> None:
    ocr = " ".join(f"word{i}" for i in range(40))
    assert recall(ocr, ocr) == 1.0
    # markdown syntax around the same words must not change the score
    assert recall(ocr, "# " + ocr + "\n") == 1.0


def test_partial_recall_counts_missing_tokens() -> None:
    ocr = " ".join(f"word{i}" for i in range(40))  # 40 tokens, measurable
    md = " ".join(f"word{i}" for i in range(30))  # 30 of 40 present
    score = recall(ocr, md)
    assert score is not None and abs(score - 0.75) < 1e-9


def test_fabrication_cannot_raise_recall() -> None:
    ocr = " ".join(f"word{i}" for i in range(40))
    md = " ".join(f"word{i}" for i in range(20)) + " invented nonsense " * 10
    score = recall(ocr, md)
    assert score is not None and score < 0.6  # still judged by OCR coverage


def test_short_ocr_is_unmeasurable() -> None:
    assert recall("tiny", "# anything") is None
    assert recall("", "# anything") is None
    # exactly at the minimum IS measurable
    ocr = " ".join(f"w{i}" for i in range(30))
    assert recall(ocr, ocr) == 1.0


def test_floor_failure_gate_semantics() -> None:
    # unmeasurable never fails
    assert floor_failure(None, 0.80) is None
    # above floor passes
    assert floor_failure(0.85, 0.80) is None
    # at the floor passes (>=)
    assert floor_failure(0.80, 0.80) is None
    # below floor fails with an actionable reason
    reason = floor_failure(0.42, 0.80)
    assert reason is not None and "42%" in reason and "80%" in reason


def test_floor_failure_reason_mentions_token_recall() -> None:
    reason = floor_failure(0.10, 0.80)
    assert reason is not None and "token recall" in reason


def test_disabled_floor_never_fails() -> None:
    assert floor_failure(0.0, 0.0) is None


def test_math_formatting_survives_tokenization() -> None:
    # LaTeX math in markdown: $E = mc^2$ — tokens survive despite markers
    tokens = tokenize("The energy $E = mc^2$ formula")
    assert "energy" in tokens and "formula" in tokens
    # the math content tokens also survive (alphanumerics)
    assert "mc" in tokens
