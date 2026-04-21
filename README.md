# Receipt Processing Agent

Local backend that accepts a folder of receipt images (or multipart uploads),
extracts fields via OCR, categorizes each receipt into an approved expense
category using a bounded DeepSeek sub-agent, aggregates totals, and streams the
full execution trajectory over Server-Sent Events while persisting a reviewable
trace to Postgres.

## Prerequisites

- Python 3.11+
- A Supabase Cloud project (connection string required — **yes, even to run the code**; see `DESIGN.md`)
- For `LLM_MODE=real`: OpenAI and DeepSeek API keys

## Setup

```bash
make install             # creates venv, installs deps
cp .env.example .env     # fill in SUPABASE_DB_URL at minimum
# activate the venv for subsequent commands:
. .venv/bin/activate
make migrate             # apply Alembic migrations to your Supabase DB
```

## Run

```bash
make run
# → uvicorn on http://0.0.0.0:8000
```

## Mock mode (no API keys needed)

```bash
LLM_MODE=mock make run
```

In `mock` mode, OCR and LLM adapters are swapped for deterministic fakes keyed
by filename / vendor. Tests always run in mock mode.

## Example — folder input

```bash
curl -N -X POST http://localhost:8000/runs/stream \
     -H "content-type: application/json" \
     -d '{"folder_path": "./assets", "prompt": "flag unusual invoices"}'
```

## Example — multipart upload

```bash
curl -N -X POST http://localhost:8000/runs/stream \
     -F "files=@assets/receipt_001.png" \
     -F "files=@assets/receipt_002.png" \
     -F "prompt=be conservative"
```

## Example SSE output

```
event: run_started
data: {"event_type":"run_started","run_id":"...","seq":1,"ts":"...","prompt":"be conservative"}

event: progress
data: {"event_type":"progress","seq":2,"step":"load_images","ts":"..."}

event: tool_call
data: {"event_type":"tool_call","tool":"load_images","args":{"folder_path":"..."}}

event: tool_result
data: {"event_type":"tool_result","tool":"load_images","result_summary":{"count":2},"duration_ms":4,"error":false}

... (per-receipt sequence) ...

event: receipt_result
data: {"event_type":"receipt_result","receipt_id":"...","status":"ok","vendor":"Uber","total":"45.67","category":"Travel", ...}

event: final_result
data: {"event_type":"final_result","total_spend":"123.45","by_category":{"Travel":"45.67"},...,"issues_and_assumptions":[]}
```

## Tests

```bash
. .venv/bin/activate
make test         # all non-e2e tests (mock mode, no network)
make test-e2e     # real API smoke test (requires LLM_MODE=real + keys)
```

## Architecture

See `DESIGN.md` (1 page) and `spec.md` (API + schemas).
High-level system spec: `specs.md`.

## Deliberate PDF deviation
SSE event name `invoice_result` is renamed `receipt_result` for terminology
consistency. All other PDF contract names are preserved. Rationale in `DESIGN.md`.
