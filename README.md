# pdf2md — Hybrid-Agentic PDF → GitHub-Flavored Markdown Converter

> **Disclaimer**
> This project is **experimental and for educational purposes only**. It is in a
> **very early stage of development** and may contain numerous issues, bugs, and rough edges.
> Expect breaking changes, incomplete features, and behavior that has only been validated
> on a small private test corpus.

Locally-run app that converts one PDF at a time into faithful GitHub-Flavored
Markdown (text, tables, formulas, lists — plus every figure extracted with AI
alt-text). Each page is routed: born-digital text pages try deterministic
extraction first (half the Cloud cost, no local OCR), gated by a single agent
verification call plus an objective token-recall floor; anything the
deterministic path cannot serve — scanned pages, figure-dense pages, pages
that fail either gate — takes the full agentic path: a vision agent reads the
rendered page, cross-checks local OCR ground truth, and self-verifies until
it passes.

Measured on a 6-document corpus with a 34-question golden set
(`eval/BENCHMARK-REPORT.md`): **94% Recall@5 vs 68% for the pymupdf4llm
baseline** — a tie on born-digital text documents, and **0% → 87.5%** on
scanned (image-only) documents, where deterministic extraction gets nothing
and the agentic path recovers full structure (headings, tables, formulas).

Output layout (many documents share one output folder without collisions):

```text
<output>/
  <docname>.md                  # top level: one glance lists all documents
  <docname>/assets/...          # this document's figures only
  <docname>/conversion-report.md
```

Image links inside each `.md` are `<docname>/assets/…` (relative, portable).
Re-converting the same document replaces only its own files.

Privacy note: page images are sent to Ollama Cloud for the agent stage. OCR
stays fully local. Deterministic-routed pages send only the page image (one
verification call).

Status: **experimental / proof of concept** — see the disclaimer above.
Requirements are normative in `docs/SRS.md`; measured quality claims live in
`eval/BENCHMARK-REPORT.md`.

---

## 1. Prerequisites (once per machine, Windows 11 native)

No WSL/virtualization. All dependencies run as native Windows services/processes.
Use regular Command Prompt (`cmd`); steps needing Administrator rights are marked **[Admin]**.

- [ ] Windows 11, Win32 long paths enabled + reboot (**[Admin]**, one of):
  - Registry: `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1` (DWORD), or
  - `gpedit.msc` → Computer Configuration → Administrative Templates → System → Filesystem → Enable Win32 long paths.
- [ ] Python 3.14.x (`python --version`) and Git (`git --version`).
  - If `python` opens the Microsoft Store: Settings → Apps → Advanced app settings → App execution aliases → turn off `python.exe` / `python3.exe`, reopen `cmd`.
- [ ] PostgreSQL 18.x native Windows service running (`postgresql-x64-18`):
  `sc query type= service state= all | findstr /I postgresql`.
  If `psql` is not recognized, add `C:\Program Files\PostgreSQL\18\bin` to your user PATH.
- [ ] Ollama responding: `curl http://localhost:11434/api/version`
  (if it fails, launch Ollama once from the Start menu). Pull local models:
  `ollama pull glm-ocr` and `ollama pull qwen3-embedding:0.6b`, then `ollama list`.
- [ ] Ollama Cloud: `ollama signin` (browser), then create an API key named
  `pdf2md-agent` in Ollama account settings → API Keys. Verify:
  `curl https://ollama.com/api/tags -H "Authorization: Bearer YOUR_KEY"` lists `glm-5.3-flash`.
  (`glm-ocr` is local-only and must never be routed to cloud.)
- [ ] Project folders outside OneDrive if possible (OneDrive locking interferes
  with long jobs). Example: `C:\projects\pdf2md` with `corpus\` and `output\` inside.

## 2. Install (per clone)

```cmd
git clone <repo-url> pdf2md
cd pdf2md
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

`requirements.txt` mirrors `pyproject.toml` (source of truth) — runtime deps
plus dev tools (pytest, ruff, mypy).

## 3. Configure

```cmd
copy .env.example .env
notepad .env
```

Fill in the two secrets (never commit `.env`, never paste keys into chat/logs):

- `OLLAMA_API_KEY` — your Ollama Cloud key.
- Database password in `DATABASE_URL` — first create DB/user (owner password
  must differ from the `postgres` superuser password):

```cmd
psql -U postgres -h localhost -c "CREATE USER pdf2md WITH PASSWORD 'CHANGE_ME_APP_PASSWORD';"
psql -U postgres -h localhost -c "CREATE DATABASE pdf2md OWNER pdf2md;"
psql -U postgres -h localhost -c "GRANT ALL PRIVILEGES ON DATABASE pdf2md TO pdf2md;"
psql -U pdf2md -h localhost -d pdf2md -c "SELECT version();"
```

All other keys have working defaults (see `docs/SRS.md` Appendix B):
`AGENT_MODEL=glm-5.3-flash`, `OCR_MODEL=glm-ocr`, `EMBED_MODEL=qwen3-embedding:0.6b`,
`RENDER_DPI=200`, `COVERAGE_THRESHOLD=95`, `COVERAGE_FLOOR_TOKENS=80`,
`HYBRID_ROUTING=false`, `MAX_PAGE_RETRIES=2`, `CHROMA_PATH=./chroma`.
Set `HYBRID_ROUTING=true` to enable deterministic-first page routing
(recommended for mixed born-digital corpora; scanned pages route agentic
automatically either way).

Then apply migrations:

```cmd
venv\Scripts\python -m alembic upgrade head
venv\Scripts\python -m alembic current
```

`current` must print the head revision (`a73cb35dc5f1 (head)`).

## 4. Test corpus

Put **at least 5 real PDFs** in `corpus\` (plus one small 2–5 page PDF for fast runs):

| # | Type | Why |
|---|---|---|
| 1 | Scanned book chapter (image-only PDF) | Pure-vision path, no text layer |
| 2 | Technical paper with formulas + a 2+ page table | Math + cross-page merging |
| 3 | Slide-deck-style PDF | Sparse layout, big figures |
| 4 | Image-heavy report | Figure extraction + alt-text at volume |
| 5 | Two-column academic paper | Reading-order / column linearization |

`eval\make_scanned_doc.py` can synthesize a scanned fixture
(`corpus\scanned_knowledge_handbook.pdf`, zero native text) if you don't have
a real scanned PDF handy — it is the corpus entry the agentic path exists for.

## 5. Run

```cmd
python app.py
```

Open http://127.0.0.1:8000, upload a PDF, choose an output folder, and watch it
convert. (`python -m src` prints a config smoke line; `python -m src --serve`
serves identically. The entry point sets `WindowsSelectorEventLoopPolicy` first —
required on Windows for asyncpg/WebSocket stability; do not reorder imports above it.)

Health check (all six must print PASS, no secrets printed):

```cmd
venv\Scripts\python -c "from src.config import load_settings; from src.health import run_all; [print(('PASS' if r.ok else 'FAIL'), r.name, '-', r.message) for r in run_all(load_settings())]"
```

## 6. Checks

```cmd
venv\Scripts\python -m ruff format --check src tests
venv\Scripts\python -m ruff check src tests
venv\Scripts\python -m mypy src
venv\Scripts\python -m pytest tests -q
```

Live-model tests (`@pytest.mark.live`) never run by default — they spend
Cloud quota. Run them explicitly only when you need live-model verification:
`python -m pytest tests -m live -o addopts=""`.

Suite: 230 unit (no network/DB/Ollama) + 46 integration (real
Postgres/Chroma/local Ollama, cloud mocked) green. Two integration tests that
need sustained local-embedding calls (`test_driver`, `test_chroma_embed`) can be
slow if Ollama is busy.

## 6a. Benchmark (retrieval quality, measured)

The `eval\` directory ships the retrieval evaluation harness used for the
headline claims:

```cmd
venv\Scripts\python eval\make_scanned_doc.py   (once: synthesizes the scanned corpus fixture)
venv\Scripts\python -X utf8 eval\run_eval.py --pipeline eval\baseline-md --pipeline eval\pdf2md-md --chunker sections
```

- `eval/goldenset.json` — 34 questions across 6 corpus PDFs (5 born-digital
  + 1 synthetic scanned doc), each with expected substrings for hit-checking.
- `--chunker sections|hierarchy` — same documents, same embedding model
  (`qwen3-embedding:0.6b`), same Chroma retrieval; only chunking differs.
- Results: Recall@5 + MRR per pipeline, per-question detail in
  `eval/results*.json`; findings and the decision analysis in
  `eval/BENCHMARK-REPORT.md`.

Headline numbers (2026-09-10 run): pdf2md 94% Recall@5 / MRR 0.64 vs
pymupdf4llm baseline 68% / 0.54 (sections chunker). Gap decomposition:
born-digital documents tie at 88%; the scanned document swings 0% → 87.5%.
Hybrid-routing A/B (same doc converted twice): identical retrieval,
~40% less wall-clock, zero local OCR on routed pages.

## 7. Troubleshooting

### Local OCR timeouts (>120 s) / rendering slow

Model spilling to CPU (8 GB VRAM machines) — known slow path; dense pages can
take minutes with no new events (the UI elapsed clock is the heartbeat). Adding a
Windows Defender exclusion for the project folder helps rendering.

### `PermissionError` on cleanup

Expected on Windows locks (open handle/Defender): the app retries with backoff
up to `MAX_CLEANUP_RETRIES`, then defers to next start. A completed job stays
completed.

### Port already in use (`[Errno 10048]`)

A previous `python app.py` is still running: close that terminal or stop the
`python.exe` process running `app.py`, then relaunch. `Ctrl+C` shuts down
cleanly (open WebSockets close quietly).

### Restart / reinitialize from scratch (Windows 11)

Zero history: no jobs, no checkpoints, no embeddings, no workspaces, no
deliverables. Stop the server first (Ctrl+C), then:

```cmd
REM 1. Empty the database (drops tables, version row, orphaned enum types)
psql -U postgres -d pdf2md -f init_db.sql

REM 2. Rebuild the schema from the migrations (single source of truth)
venv\Scripts\python -m alembic upgrade head
venv\Scripts\python -m alembic current
```

```cmd
REM 3. Delete generated state (keep mdtopdf.py, corpus\, .env, src\)
rmdir /s /q workspace chroma
del /q output\*.md output\*.pdf
for /d %i in (output\*) do rmdir /s /q "%i"
dir output
```

`output\` should show only `mdtopdf.py` afterward. `corpus\` (inputs), `.env`
(keys), and `src\` are never touched. If `upgrade head` ever fails with
`type "jobstatus" already exists`, re-run `init_db.sql` and upgrade again.

```cmd
REM 4. Launch and verify the clean slate
python app.py
```

Expect: no "incomplete job" warnings at startup, empty History, health
all-green. Convert the `corpus\` PDFs one at a time through the UI.

Never delete `corpus\`, `.env`, or `chroma\` while the server runs (stop it
first — Windows file locks).

### Symptom → fix quick table

| Symptom | Fix |
|---|---|
| `python` opens Microsoft Store | Disable `python.exe`/`python3.exe` app execution aliases |
| `psql` not recognized | Add `C:\Program Files\PostgreSQL\18\bin` to user PATH |
| Ollama `curl` fails | Launch Ollama once from Start menu, retry |
| `ollama list` missing models | `ollama pull glm-ocr` + `ollama pull qwen3-embedding:0.6b` |
| Cloud key rejected | Recheck `OLLAMA_API_KEY` in `.env`; direct-check `/api/tags` |

## Docs

- `docs/SRS.md` — requirements (normative)
- `eval/BENCHMARK-REPORT.md` — measured retrieval benchmark + routing A/B findings
