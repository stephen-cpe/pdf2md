"""Application entry point.

The event-loop policy call below MUST remain the first executable code in
this module — before any project imports or async initialization
(WindowsSelectorEventLoopPolicy; asyncpg/WebSocket connections fail
randomly on Windows without it). Enforced by
tests/unit/test_event_loop_policy.py — do not reorder.
"""

import asyncio
import warnings

# Required first (asyncpg + WebSocket stability on Windows).
# Deprecated advisory-only in 3.14 (removal slated 3.16) — the call
# still takes effect, so keep it and silence just this warning. Revisit on 3.16.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def main() -> None:
    """Serve the app (--serve) or print a config smoke line (default)."""
    import sys

    if "--serve" in sys.argv[1:]:
        _serve()
        return
    from src.config import load_settings

    settings = load_settings()
    print(f"pdf2md OK (agent={settings.AGENT_MODEL}, ocr={settings.OCR_MODEL})")


def _serve() -> None:
    """Run the FastAPI app (UI + API)."""
    import uvicorn

    from src.api.app import create_app
    from src.config import load_settings

    settings = load_settings()
    uvicorn.run(
        create_app(settings),
        host=settings.BIND_HOST,
        port=settings.PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
