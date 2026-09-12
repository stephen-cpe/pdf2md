"""Unit tier: GFM normalization (math safety) + lint classification."""

import pytest

from src.pipeline.gfm import classify, github_math_safe, lint_markdown, normalize_markdown


def test_normalize_idempotent_and_safe() -> None:
    messy = "# T\n\nSome   spaced    text.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n*em* **strong**\n"
    once = normalize_markdown(messy)
    assert normalize_markdown(once) == once
    compact = once.replace(" ", "")
    for token in ("*em*", "**strong**", "|1|2|"):
        assert token in compact


def test_normalize_preserves_inline_math_verbatim() -> None:
    """Regression: mdformat has no math extension and used to escape LaTeX.

    `\\text{old\\_way}` became `\\\\text{old_way}` (invalid math: a `\\\\`
    line break plus a misplaced `_`), which GitHub rejects with
    "'_' allowed only in math mode".
    """
    source = r"The value of $C_{\text{old\_way}}$ is shown in Table 5(d)."
    assert normalize_markdown(source).rstrip("\n") == source


def test_normalize_preserves_display_math_verbatim() -> None:
    source = (
        r"$$\text{Cost savings} = \sum C_{\text{old\_way}} "
        r"- \sum C_{\text{new\_way}} - C_{\text{del}} - C_{\text{ref}}$$"
    )
    assert normalize_markdown(source).rstrip("\n") == source


def test_normalize_preserves_backslash_commands() -> None:
    source = r"$$\text{Cost savings} = \sum C - \{C_{\text{org}} + C_{\text{cab}}\}$$"
    assert normalize_markdown(source).rstrip("\n") == source


def test_normalize_preserves_multiline_display_math() -> None:
    source = "$$\n\\begin{aligned}\na &= b \\\\\nc &= d\n\\end{aligned}\n$$"
    assert source in normalize_markdown(source)


def test_normalize_handles_plain_dollar_text() -> None:
    """Unpaired dollar signs in prose are not math and stay untouched."""
    source = "Cost is $5 and value is $10 today."
    assert normalize_markdown(source).rstrip("\n") == source


def test_normalize_still_reformats_around_math() -> None:
    source = "# Math\n\nText   before $x_1$ and   after.\n"
    out = normalize_markdown(source)
    assert "$x_1$" in out
    assert "Text before $x_1$ and after." in out


@pytest.mark.parametrize(
    "math",
    [
        r"$C_{\text{old\_way}}$",
        r"$\sum_{i=1}^{n} x_i$",
        r"$\{a, b\}$",
        r"$\frac{1}{2}$",
        r"$a \to b$",
    ],
)
def test_normalize_never_escapes_math_spans(math: str) -> None:
    out = normalize_markdown(f"Equation {math} inline.")
    assert math in out, out


def test_lint_clean_and_classify() -> None:
    assert lint_markdown("# T\n\nClean paragraph.\n") == []
    verdict = classify(False, ["missing asset"], ["rule hit"])
    assert verdict.hard_fail and verdict.reasons == ["missing asset"]
    assert classify(True, [], ["rule hit"]).hard_fail is False


# --- github_math_safe: output transform for GitHub's math unescaping ---
#
# GitHub's Markdown pipeline strips one backslash-escape level from `$...$`
# before KaTeX: `\text{old\_way}` -> `\text{old_way}` (bare `_` ->
# "'_' allowed only in math mode"), `\{` -> `{`. Double-escaping is not the
# fix (GitHub then drops the formula out of math mode entirely). Validated
# against POST https://api.github.com/markdown + KaTeX: `\rule` draws the
# underscore and `\lbrace`/`\rbrace` survive.

UNDERSCORE_RULE = r"\rule[-0.05em]{0.35em}{0.5pt}"


def test_github_math_safe_rewrites_underscore_in_text() -> None:
    source = r"Equation $C_{\text{old\_way}}$ inline."
    out = github_math_safe(source)
    assert r"\_" not in out
    assert out == rf"Equation $C_{{\text{{old}}{UNDERSCORE_RULE}\text{{way}}}}$ inline."


def test_github_math_safe_handles_nested_and_repeated_underscores() -> None:
    source = r"$$\sum C_{\text{a\_b}} + C_{\text{c\_d}}$$"
    out = github_math_safe(source)
    assert r"\_" not in out
    assert out.count(UNDERSCORE_RULE) == 2


def test_github_math_safe_rewrites_escaped_braces() -> None:
    source = r"$$x = \{a + b\}$$"
    out = github_math_safe(source)
    assert r"\{" not in out and r"\}" not in out
    assert r"\lbrace a + b \rbrace" in out


def test_github_math_safe_leaves_underscore_outside_text() -> None:
    """Subscript underscores (`C_{old}`) are valid math and untouched."""
    source = r"$C_{old} + x_1$"
    assert github_math_safe(source) == source


def test_github_math_safe_ignores_prose() -> None:
    source = "A snake_case word and a path C:\\Users\\x and $x_1$."
    assert github_math_safe(source) == source


def test_github_math_safe_is_idempotent() -> None:
    source = r"$C_{\text{old\_way}}$ and $\{a\}$"
    once = github_math_safe(source)
    assert github_math_safe(once) == once


def test_github_math_safe_preserves_non_math_bytes() -> None:
    import re

    math_span = re.compile(r"\$\$.*?\$\$|\$[^$\n]+?\$", re.DOTALL)
    source = "# Title\n\nProse with $C_{\\text{old\\_way}}$ here.\n\n- list item\n"
    out = github_math_safe(source)
    assert math_span.sub("", out) == math_span.sub("", source)
