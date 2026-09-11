"""Root launcher: `python app.py` serves the pdf2md web UI.

Classic venv workflow (from the project root)::

    python -m venv venv
    venv\\Scripts\\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python app.py

The event-loop policy call below MUST stay the first executable code in this
file — before the path bootstrap and any project imports
(WindowsSelectorEventLoopPolicy; asyncpg/WebSocket connections fail randomly
on Windows without it). Same rule as src/__main__.py.
"""

import asyncio
import warnings

# Required first (asyncpg + WebSocket stability on Windows).
# Deprecated advisory-only in 3.14 (removal slated 3.16) — the call
# still takes effect, so keep it and silence just this warning. Revisit on 3.16.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.__main__ import _serve  # noqa: I001  (path bootstrap must precede project imports)


if __name__ == "__main__":
    _serve()
