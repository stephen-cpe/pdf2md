"""The entry point's first executable code MUST set the
Windows selector event-loop policy before any other imports/async init.

Two layers: static (AST order in __main__.py) + runtime (policy actually in
effect in a fresh interpreter running the entry point).
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[2] / "src" / "__main__.py"


def _top_level_nodes() -> list[ast.stmt]:
    return ast.parse(ENTRYPOINT.read_text(encoding="utf-8")).body


def _is_policy_call(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "set_event_loop_policy"
        and bool(node.value.args)
        and isinstance(node.value.args[0], ast.Call)
        and isinstance(node.value.args[0].func, ast.Attribute)
        and node.value.args[0].func.attr == "WindowsSelectorEventLoopPolicy"
    )


def test_policy_call_is_first_executable_code() -> None:
    """set_event_loop_policy(WindowsSelectorEventLoopPolicy()) comes before
    any project import, any other call, and any def/class block — either
    bare or inside the warnings.catch_warnings() silencer."""
    policy_seen = False
    problems: list[str] = []
    for node in _top_level_nodes():
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in getattr(node, "names", []):
                mod = getattr(name, "name", "")
                if mod.startswith("src"):
                    problems.append(f"project import before policy: {mod}")
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src"):
                problems.append("project import before policy")
            continue
        if isinstance(node, ast.With):
            # Allowed only as the deprecation-warning silencer around the call:
            # every statement must be a warnings.* call except the last,
            # which must be the policy call.
            if not node.body or not _is_policy_call(node.body[-1]):
                problems.append("unexpected with-block before the policy call")
                break
            for stmt in node.body[:-1]:
                if not (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Attribute)
                    and isinstance(stmt.value.func.value, ast.Name)
                    and stmt.value.func.value.id == "warnings"
                ):
                    problems.append("unexpected with-block before the policy call")
                    break
            policy_seen = True
            break
        if _is_policy_call(node):
            policy_seen = True
            break
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            problems.append("a call precedes the event-loop policy call")
            break
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If)):
            break
    assert not problems, problems
    assert policy_seen, "no set_event_loop_policy(WindowsSelectorEventLoopPolicy()) found first"


def test_policy_in_effect_at_runtime() -> None:
    """Fresh interpreter importing the entry point gets the policy."""
    code = (
        "import asyncio, src.__main__; "
        "assert isinstance(asyncio.get_event_loop_policy(), "
        "asyncio.WindowsSelectorEventLoopPolicy), 'policy not set'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=60)


def test_entrypoint_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """python -m src exits clean. Fresh-clone safe: fake secrets come
    from env vars, so no .env file is required (env vars are inherited
    by the subprocess; a developer's real .env, when present, also works)."""
    for key, value in (
        ("OLLAMA_API_KEY", "unit-test-fake-key"),
        ("DATABASE_URL", "postgresql+asyncpg://u:unit-test@localhost:5432/db"),
    ):
        monkeypatch.setenv(key, value)
    proc = subprocess.run(
        [sys.executable, "-m", "src"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert proc.returncode == 0, proc.stderr
    assert "pdf2md OK" in proc.stdout
