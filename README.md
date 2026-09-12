# pdf2md — PDF → GitHub-Flavored Markdown, with diagrams reinterpreted as Mermaid

## Disclaimer
> This project is **experimental and for educational purposes only**. It is in a
> **very early stage of development** and may contain numerous issues, bugs, and rough edges.
> Expect breaking changes, incomplete features, and behavior that has only been validated
> on a small private test corpus.

Locally-run app that converts one PDF at a time into faithful GitHub-Flavored
Markdown (text, tables, formulas, lists) **and reinterprets its figures —
diagrams, flowcharts, charts, graphs, schematics — as Mermaid source** wherever
that can be done faithfully. Each page is transcribed by a vision agent
(`glm-5.3-flash` on Ollama Cloud) cross-checked against local OCR ground truth
(`glm-ocr`), gated by a self-verification pass and an objective token-recall
floor. Then every flagged figure region is cropped, optionally grounded in the
text inside it, and sent to a dedicated diagram→Mermaid conversion stage that is
validated (type allowlist + structural checks) and vision-verified before it is
trusted.

The Mermaid code will not be perfect today. The models are improving, and the
design assumes they will keep improving: reinterpretation quality is the metric
the project now optimizes, and every fallback keeps the original figure so no
information is ever silently lost.

Output layout (many documents share one output folder without collisions):

```text
<output>/
  <docname>.md                  # top level: one glance lists all documents
  <docname>/assets/...          # this document's figures only
  <docname>/conversion-report.md
```

Image links inside each `.md` are `<docname>/assets/…` (relative, portable).
Re-converting the same document replaces only its own files.

Privacy note: page images **and cropped figure regions** are sent to Ollama
Cloud for the agent and diagram stages. OCR stays fully local.

Status: **experimental / proof of concept** — see the disclaimer above.
Requirements are normative in `docs/SRS.md`.

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
  (if it fails, launch Ollama once from the Start menu). Pull the local OCR model:
  `ollama pull glm-ocr`, then `ollama list`.
- [ ] Ollama Cloud: `ollama signin` (browser), then create an API key named
  `pdf2md-agent` in Ollama account settings → API Keys. Verify:
  `curl https://ollama.com/api/tags -H "Authorization: Bearer YOUR_KEY"` lists `glm-5.3-flash`.
  (`glm-ocr` is local-only and must never be routed to cloud.)
- [ ] Project folders outside OneDrive if possible (OneDrive locking interferes
  with long jobs). Example: `C:\projects\pdf2md` with `corpus\` and `output\` inside.

## 2. Install (per clone)

```cmd
git clone https://github.com/stephen-cpe/pdf2md.git pdf2md
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
`AGENT_MODEL=glm-5.3-flash`, `OCR_MODEL=glm-ocr`, `RENDER_DPI=200`,
`COVERAGE_THRESHOLD=95`, `COVERAGE_FLOOR_TOKENS=80`, `MAX_PAGE_RETRIES=2`.

Diagram→Mermaid (the primary capability) is on by default and config-gated:

- `DIAGRAM_TO_MERMAID=true` — enable figure reinterpretation.
- `DIAGRAM_ALLOWED_TYPES` — comma-separated Mermaid type allowlist.
- `DIAGRAM_MIN_CONFIDENCE=80` — converter confidence needed to accept.
- `DIAGRAM_VERIFY=true` — vision-verify each candidate against the crop.
- `DIAGRAM_FALLBACK=both` — `image` | `table` | `both` (charts get tables).
- `DIAGRAM_KEEP_IMAGE=true` — keep the original figure under a converted Mermaid.

Then apply migrations:

```cmd
venv\Scripts\python -m alembic upgrade head
venv\Scripts\python -m alembic current
```

`current` must print the head revision (`b84dc46ed6a2 (head)`).

## 4. Test corpus

`corpus\` is gitignored, so create it first, then put **at least 5 real
PDFs** inside (plus one small 2–5 page PDF for fast runs):

| # | Type | Why |
|---|---|---|
| 1 | Scanned book chapter (image-only PDF) | Pure-vision path, no text layer |
| 2 | Technical paper with formulas + a 2+ page table | Math + cross-page merging |
| 3 | Slide-deck-style PDF | Sparse layout, big figures |
| 4 | Image-heavy report | Figure extraction + Mermaid at volume |
| 5 | Two-column academic paper | Reading-order / column linearization |

## 5. Run

```cmd
python app.py
```

Open http://127.0.0.1:8000, upload a PDF, choose an output folder, and watch it
convert. (`python -m src` prints a config smoke line; `python -m src --serve`
serves identically. The entry point sets `WindowsSelectorEventLoopPolicy` first —
required on Windows for asyncpg/WebSocket stability; do not reorder imports above it.)

Health check (all five must print PASS, no secrets printed):

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

## 7. Diagram → Mermaid (primary capability)

For every figure region the page agent flags:

1. **Crop** the region from the rendered page.
2. **Ground** it in native text inside the region (best effort).
3. **Convert** with a dedicated Cloud call (`DIAGRAM_TEMPLATE`).
4. **Validate** — Mermaid type on the first line, allowlisted, non-empty body,
   no leftover pipeline markers.
5. **Verify** — a second Cloud call compares the Mermaid source against the
   crop (`DIAGRAM_VERIFY_TEMPLATE`); fabricated nodes/edges/values fail it.
6. **Emit** the tiered representation:

| Tier | When | Output at the token site |
|---|---|---|
| Mermaid | validated + verified | ` ```mermaid ` block + collapsible original figure |
| Table + image | data chart, or Mermaid fidelity fails | OCR-grounded GFM table + image |
| Image + alt | photo/illustration/map, or crop fails | image + alt + caption + long description |

Nothing is silently dropped: a figure that cannot be represented still lands as
an asset link, and the conversion report lists per-figure status and the overall
conversion rate.

## 8. Troubleshooting

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

Stop the server first (Ctrl+C), then run all steps in order from the project
root. Step 1 drops the tables and only step 2 rebuilds them.

```cmd
psql -U postgres -h localhost -d pdf2md -f init_db.sql
venv\Scripts\python -m alembic upgrade head
venv\Scripts\python -m alembic current
```

`current` must print `b84dc46ed6a2 (head)`. Empty output means the upgrade
did not apply — do not continue; re-run the upgrade and read its error.

```cmd
rmdir /s /q workspace
del /q output\*.md output\*.pdf
for /d %i in (output\*) do rmdir /s /q "%i"
```

This deletes generated state only — `corpus\` (your PDFs), `.env` (your keys),
and `src\` are never touched. If `upgrade head` ever fails with
`type "jobstatus" already exists`, the old enum types survived the reset:
re-run `init_db.sql` and upgrade again.

```cmd
python app.py
```

### Symptom → fix quick table

| Symptom | Fix |
|---|---|
| `python` opens Microsoft Store | Disable `python.exe`/`python3.exe` app execution aliases |
| `psql` not recognized | Add `C:\Program Files\PostgreSQL\18\bin` to user PATH |
| Ollama `curl` fails | Launch Ollama once from Start menu, retry |
| `ollama list` missing model | `ollama pull glm-ocr` |
| Cloud key rejected | Recheck `OLLAMA_API_KEY` in `.env`; direct-check `/api/tags` |
| `relation "jobs" does not exist` after a reset | Re-run `alembic upgrade head` and confirm `alembic current` prints `b84dc46ed6a2 (head)` before starting the app |

## Docs

- `docs/SRS.md` — requirements (normative)
