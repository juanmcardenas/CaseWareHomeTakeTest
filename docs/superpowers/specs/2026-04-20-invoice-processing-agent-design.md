# Invoice Processing Agent — Design

**Date:** 2026-04-20
**Status:** Approved by user; ready for implementation planning.

## 1. Context & scope

Build a local backend service (Python + FastAPI) that ingests a folder of invoice images or a multipart upload, extracts per-invoice structured data via OCR, categorizes each invoice into an approved expense category, aggregates totals, and streams the execution trajectory via Server-Sent Events. Each run produces a reviewable trace with step boundaries, tool calls, planner decisions, errors, and timing.

This spec is the output of a brainstorming session that resolved ambiguities in the input `specs.md` against the take-home PDF (`Take-Home Test - Software Development Manager, AI (4) (1).pdf`).

### Source-of-truth reconciliation

- **`specs.md`** is the user's internal stack commitment: Supabase, SQLAlchemy, Alembic, Langfuse, DeepSeek-chat, OpenAI OCR, LangChain + LangGraph, hexagonal architecture.
- **Take-home PDF** is the evaluation contract: 2–3h timebox, local execution, `POST /runs/stream` SSE endpoint, 4–6 tool registry, reviewable trace deliverable, mock mode recommended.
- Where they conflict, the PDF's *contract* requirements win; the `specs.md` *stack* wins when it doesn't violate the PDF.

### Explicit timebox risk

The PDF imposes a 2–3h timebox with "do not over-engineer". The `specs.md` stack is heavy relative to that budget. The user has opted to keep the full stack and a depth-over-breadth scope (per decision Q1). This spec records that decision and designs accordingly: shallow-but-correct hexagonal skeleton, single Alembic revision, Langfuse instrumented via one decorator, write-through persistence as the *primary* trace mechanism (Langfuse is secondary, not on the critical path).

### Supabase reviewer-runnability risk

The user chose Supabase Cloud only (decision Q5). The PDF says "local execution only". Reviewer will need a Supabase project + connection string to run the code. Surfaced prominently in `README.md` setup instructions and `DESIGN.md`.

## 2. Decision log

| # | Question | Decision | Rationale |
|---|---|---|---|
| Q1 | Scope & ambition | Depth over breadth | Demonstrates architecture; ships in timebox |
| Q2 | Agent vs deterministic | Hybrid: deterministic graph + bounded sub-agent | Real planner decisions where judgment lives; deterministic everywhere else |
| Q3 | Sub-agent scope | Medium: `{category, confidence, notes, issues[]}` per invoice | Produces run-level "Issues & Assumptions" naturally; 1 LLM call/invoice |
| Q4 | Multi-invoice processing | Sequential | Cleanest trajectory for walkthrough; simpler trace story |
| Q5 | Supabase deployment | Supabase Cloud only | User preference; reviewer friction recorded as risk |
| Q6 | Mock mode | Single `LLM_MODE=mock` env switch | Removes LLM-key friction; deterministic tests |
| Q7 | Persistence model | Write-through (event → `traces` insert) | Honest "reviewable trace" + post-run DB inspection |
| Q8 | Prompt steering | Free-form text injected into sub-agent system prompt | Natural, no fixed menu, full trajectory capture |

## 3. Architecture

Hexagonal (Ports & Adapters). Dependencies point inward. Three layers.

### Domain (`src/domain/`)

Pure Python. Pydantic models (`Invoice`, `NormalizedInvoice`, `Categorization`, `Issue`, `Report`, `Aggregates`, `AllowedCategory` enum). Pure functions for normalization and aggregation. No I/O, no framework imports.

### Application (`src/application/`)

- **LangGraph state machine** (`graph.py`) — deterministic outer pipeline. Sequential loop over invoices. Fixed edges: `load_images → [for each invoice: ocr → normalize → categorize_with_subagent → invoice_result] → aggregate → generate_report`.
- **Tool registry** (`tool_registry.py`) — 6 tools, the only way invoice data is touched. Each tool is a thin wrapper decorated with `@traced_tool` (event emission + Langfuse span + timing + error classification).
- **Categorization sub-agent** (`subagent.py`) — one DeepSeek call per invoice. Input: normalized fields + allowed categories + user prompt injected into system message. Output: strict Pydantic `Categorization`. The only place judgment lives.
- **Ports** (`ports.py`) — abstract interfaces: `OCRPort`, `LLMPort`, `ImageLoaderPort`, `ReportRepositoryPort`, `TraceRepositoryPort`, `EventBusPort`, `TracerPort`.
- **Events** (`events.py`) — Pydantic SSE event models.
- **EventBus** (`event_bus.py`) — in-process async pub/sub; fan-out to HTTP response, trace writer, Langfuse adapter, JSON logger.

### Infrastructure (`src/infrastructure/`)

- `ocr/` — `OpenAIOCRAdapter`, `MockOCRAdapter` (swapped by `LLM_MODE`)
- `llm/` — `DeepSeekLLMAdapter`, `MockLLMAdapter` (same switch)
- `images/` — `LocalFolderImageLoader`, `UploadImageLoader`
- `db/` — SQLAlchemy engine, ORM rows (`ReportRow`, `ReceiptRow`, `TraceRow`), repositories
- `tracing/` — `LangfuseTraceAdapter`, `JSONLogsTraceAdapter` (both active)
- `http/` — FastAPI app, `sse.py` helper, `routes_runs.py`

Wiring happens exclusively in `composition_root.py`.

### Tool registry (6)

1. `load_images(input)` → `[ImageRef]`
2. `extract_invoice_fields(image_ref)` → `RawInvoice` (OpenAI OCR)
3. `normalize_invoice(raw)` → `NormalizedInvoice` (pure domain)
4. `categorize_invoice(normalized, prompt, allowed)` → `Categorization` (DeepSeek sub-agent)
5. `aggregate(invoices)` → `Aggregates`
6. `generate_report(aggregates, invoices, issues)` → `Report` (deterministic)

## 4. Module layout

```
src/
├── domain/
│   ├── models.py
│   ├── aggregation.py
│   └── normalization.py
├── application/
│   ├── ports.py
│   ├── events.py
│   ├── event_bus.py
│   ├── tool_registry.py
│   ├── subagent.py
│   └── graph.py
├── infrastructure/
│   ├── ocr/ {openai_adapter.py, mock_adapter.py}
│   ├── llm/ {deepseek_adapter.py, mock_adapter.py}
│   ├── images/ {folder_loader.py, upload_loader.py}
│   ├── db/ {engine.py, models.py, repositories.py}
│   ├── tracing/ {langfuse_adapter.py, json_logs_adapter.py}
│   └── http/ {app.py, sse.py, routes_runs.py}
├── config.py                  # Pydantic Settings
├── composition_root.py
└── main.py

migrations/                    # Alembic, one initial revision
tests/ {domain, application, infrastructure, e2e, fakes, fixtures}
assets/                        # 4–5 sample invoice images (synthetic/public)
transcripts/                   # AI coding tool interaction logs
README.md, spec.md, DESIGN.md, AGENTS.md, .env.example
```

**Boundary property**: `domain` imports nothing from `application` or `infrastructure`. `application` imports only from `domain` and `application/ports.py`. All wiring in `composition_root.py`.

## 5. Data flow & SSE trajectory

### Request lifecycle

```
POST /runs/stream
  → validate (multipart OR {folder_path, prompt})
  → create report_id, insert reports row (status=running)
  → open EventSourceResponse
  → run LangGraph in background task
  → graph emits events to EventBus
  → SSE generator yields to client
  → TraceRepository write-throughs
  → Langfuse spans open/close
  → JSON logger always on
```

### Event sequence (happy path, N invoices, sequential)

```
run_started
progress(load_images)
tool_call(load_images)
tool_result(load_images, count=N)

# Per invoice i of N:
progress(ocr, invoice_id, i, n)
tool_call(extract_invoice_fields, invoice_id)
tool_result(extract_invoice_fields, invoice_id, summary)
progress(normalize, invoice_id)
tool_call(normalize_invoice, invoice_id)
tool_result(normalize_invoice, invoice_id)
progress(categorize, invoice_id)
tool_call(categorize_invoice, invoice_id, prompt_excerpt, allowed)
tool_result(categorize_invoice, invoice_id, category, confidence, issue_count)
invoice_result(invoice_id, status, vendor, invoice_date, invoice_number,
               total, category, confidence, notes, issues[])
               # status: "ok" | "error". On error, downstream fields may be null.

# Post-loop:
progress(aggregate)
tool_call(aggregate)
tool_result(aggregate, total_spend, by_category)
progress(generate_report)
tool_call(generate_report)
tool_result(generate_report)

final_result(run_id, total_spend, by_category, invoices[], issues_and_assumptions[])
```

### Payload conventions

- Every event carries: `run_id`, monotonic `seq`, ISO-8601 `ts`, `event_type`.
- Invoice-scoped events carry: `invoice_id`.
- `tool_result` events carry a bounded `result_summary`. Full payload lives in `traces.payload` / `receipts.raw_ocr` / `receipts.normalized`.
- SSE stream is the bounded view. DB is authoritative.

### Prompt plumbing

User prompt is stored on `reports.prompt`, echoed in `run_started`, and injected verbatim into the categorization sub-agent's system message. Sub-agent's reasoning summary is returned in its response and persisted on `receipts.notes` / `traces`.

## 6. Persistence

### Schema

Three tables: `reports`, `receipts`, `traces`. Alembic, single initial revision.

```sql
reports (
  id            UUID PRIMARY KEY,
  started_at    TIMESTAMPTZ NOT NULL,
  finished_at   TIMESTAMPTZ NULL,
  status        TEXT NOT NULL,           -- 'running' | 'succeeded' | 'failed'
  prompt        TEXT NULL,
  input_kind    TEXT NOT NULL,           -- 'upload' | 'folder'
  input_ref     TEXT NULL,
  invoice_count INT NULL,
  total_spend   NUMERIC(14,2) NULL,
  by_category   JSONB NULL,              -- {"Travel": 123.45, ...}
  issues        JSONB NULL,              -- run-level issues & assumptions
  error         TEXT NULL
);

receipts (
  id             UUID PRIMARY KEY,
  report_id      UUID NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
  seq            INT NOT NULL,            -- 1..N within the run
  source_ref     TEXT NOT NULL,           -- filename or image_ref
  vendor         TEXT NULL,
  invoice_date   DATE NULL,
  invoice_number TEXT NULL,
  total          NUMERIC(14,2) NULL,
  currency       TEXT NULL,
  category       TEXT NULL,               -- AllowedCategory
  confidence     NUMERIC(3,2) NULL,       -- 0.00–1.00
  notes          TEXT NULL,
  issues         JSONB NULL,              -- per-invoice issue list
  raw_ocr        JSONB NULL,              -- full OCR payload
  normalized     JSONB NULL,              -- full normalized payload
  status         TEXT NOT NULL,           -- 'ok' | 'error'
  error          TEXT NULL,
  created_at     TIMESTAMPTZ NOT NULL
);

traces (
  id          BIGSERIAL PRIMARY KEY,
  report_id   UUID NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
  invoice_id  UUID NULL,                   -- correlation only, NOT a FK
                                           --   (traces are written before receipts rows)
  seq         INT NOT NULL,                -- monotonic within the run
  event_type  TEXT NOT NULL,               -- run_started | progress | tool_call |
                                           --   tool_result | invoice_result |
                                           --   final_result | error
  step        TEXT NULL,                   -- e.g., 'ocr' | 'normalize'
  tool        TEXT NULL,                   -- tool name if event_type = tool_*
  payload     JSONB NOT NULL,              -- full event payload
  duration_ms INT NULL,                    -- on tool_result
  created_at  TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_traces_report_seq   ON traces (report_id, seq);
CREATE INDEX idx_traces_report_type  ON traces (report_id, event_type);
CREATE INDEX idx_receipts_report_seq ON receipts (report_id, seq);
```

### Write-through semantics

- `reports` row inserted *before* the SSE stream opens (run has an ID even on early failure).
- Every event published to `EventBus` is persisted to `traces` by the `TraceRepository` subscriber inside the emit path. If DB insert fails, we log and keep streaming — trace writes never kill the run.
- An `invoice_id` (UUID) is generated at the start of each per-invoice iteration and used as (a) the correlation ID on all events/trace rows for that invoice and (b) the primary key of the `receipts` row. `traces.invoice_id` is therefore written before the `receipts` row exists, which is why it is a correlation column rather than a FK.
- `receipts` inserted at `invoice_result`. Primary key equals the pre-generated `invoice_id`.
- `reports` updated at `final_result` with `finished_at`, `status`, `total_spend`, `by_category`, `issues`. On unrecoverable error: same update with `status='failed'` and `error`.
- Each write is its own short transaction. No run-wide transaction.
- FK `ON DELETE CASCADE` so cleaning a report cleans its descendants.

### Replay

A reviewer replays a run via `SELECT payload FROM traces WHERE report_id = ? ORDER BY seq` — exactly what the SSE stream emitted.

## 7. Error handling

### Three severity bands

**Band A — Invoice-level recoverable (continue run)**
- Triggers: OCR timeout, empty OCR, LLM timeout/429, invalid category from LLM, date/total parse failure, file unreadable.
- Response: try/except around the failing tool call. Emit `tool_result` with `error: true`. Append `Issue(severity="invoice_error", code, message)` to that invoice's issues. Skip any *downstream pipeline steps* for that invoice (e.g., OCR failed → no normalize, no categorize). **Still emit an `invoice_result` event** with whatever fields are known, `status="error"`, and the issue list — this guarantees every invoice has a terminal event in the stream. Insert the `receipts` row at that `invoice_result` with `status='error'`. Exclude from aggregation.
- Retries: **one** retry with exponential backoff (1s, 2s) on network-class errors only (timeout, 429, 5xx). Retry attempt emits `tool_call` with `attempt: 2`. No retries on validation errors.

**Band B — Invoice-level soft warnings (continue, annotate, include)**
- Triggers: missing invoice number, ambiguous currency, total doesn't match line-item sum, OCR confidence < 0.6, sub-agent confidence < 0.5.
- Response: sub-agent emits these in its `issues[]`. Flow into `receipts.issues` and run-level `issues_and_assumptions`. Severity `"warning"`. Invoice is counted in aggregation.

**Band C — Run-level unrecoverable (fail run)**
- Triggers: DB unreachable at startup, no images found in input, invalid request, 100% of invoices failed at invoice level.
- Response: emit terminal `error` event. Update `reports.status='failed'`. Close stream.

### Boundary rules

- `EventBus.publish` never raises.
- Subscribers (TraceRepository, Langfuse, logger) swallow and log their own failures. The stream must not die because a trace row failed to insert.
- All tool wrappers share one `@traced_tool` decorator (Langfuse span + timing + try/except + event emission).
- Sub-agent output is validated against a strict Pydantic model. Schema mismatch → `invalid_category` (Band A).
- Timeouts: OCR 30s, LLM 20s, tool wall-clock ≤ 45s. Configurable via env.

### `issues_and_assumptions` composition

- All `Issue` objects from all invoices with `{invoice_id, severity, code, message}`.
- Plus fixed run-level assumptions: "only {jpg,jpeg,png,webp} considered", "totals assume USD when currency absent", "invoices with OCR failures excluded from aggregation".
- Empty list is valid.

## 8. Testing strategy

Four layers, prioritized by value-per-minute given the 2–3h timebox.

**Layer 1 — Domain** (`tests/domain/`, ~15 tests, <1s)
- `test_aggregation.py`: totals, by-category math, rounding, empty-list.
- `test_normalization.py`: date variants, currency normalization, invalid-input raises.
- `test_models.py`: Pydantic validators, invalid `AllowedCategory`, confidence bounds, "Other" requires a note.

**Layer 2 — Application** (`tests/application/`, ~8 tests, <5s)
- `test_graph_happy_path.py`: mock adapters, assert event sequence matches spec.
- `test_graph_invoice_error.py`: 1 of 3 OCR fails; run completes; invoice 2 has error; aggregation excludes it.
- `test_graph_run_error.py`: loader returns 0 images; terminal error; `reports.status='failed'`.
- `test_subagent_prompt_steering.py`: user prompt is injected into the fake LLM's system message.
- `test_tool_registry.py`: each tool emits exactly one `tool_call` + one `tool_result`; timing populated.

**Layer 3 — Infrastructure** (`tests/infrastructure/`, ~4 tests)
- `test_runs_stream_contract.py`: FastAPI TestClient, multipart with 2 fake images, assert terminal `final_result` fields.
- `test_runs_stream_folder.py`: JSON body with folder path.
- `test_traces_write_through.py`: after a mock run, `traces` rows match emitted events.
- `test_repositories.py`: CRUD round-trip. **Depends on Supabase cloud reachability**; marked and skippable.

**Layer 4 — E2E** (`tests/e2e/`, `@pytest.mark.e2e`, skipped by default)
- `test_real_run_smoke.py`: requires `LLM_MODE=real` + real keys. Runs one invoice from `assets/`. Validates mock/real contract parity.

**Conventions**: `pytest` + `pytest-asyncio`, per-layer `conftest.py`, mock adapters in `tests/fakes/` (test-only), fixture images in `tests/fixtures/`.

**Not tested in this build**: Langfuse integration (spot-check manually), SSE reconnect semantics, concurrency stress, SQL perf.

**Slip order if timebox bites**: Layer 3 repository tests slip first, then Layer 3 write-through verification, then Layer 2 tool-registry contract. Layer 1 never slips.

## 9. Configuration

`.env.example`:

```
# Mode
LLM_MODE=mock                 # mock | real

# Database (Supabase cloud)
SUPABASE_DB_URL=postgresql+psycopg://user:pass@host:6543/postgres

# LLM / OCR (required when LLM_MODE=real)
OPENAI_API_KEY=
DEEPSEEK_API_KEY=

# Observability
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com

# Assets
ASSETS_DIR=./assets

# Timeouts (seconds) + upload limits
OCR_TIMEOUT_S=30
LLM_TIMEOUT_S=20
TOOL_WALL_TIMEOUT_S=45
MAX_FILE_SIZE_MB=10
MAX_FILES_PER_RUN=25
ALLOWED_EXTENSIONS=jpg,jpeg,png,webp
```

Loaded via Pydantic `Settings`. Missing required keys in `real` mode fail fast at startup.

## 10. Deliverables mapping (PDF → repo)

| PDF requirement | Repo artifact |
|---|---|
| HTTP streaming endpoint `POST /runs/stream` | `src/infrastructure/http/routes_runs.py` |
| SSE events (run_started, progress, tool_call, tool_result, invoice_result, final_result, error) | `src/application/events.py` + `event_bus.py` |
| Tool registry (4–6) | `src/application/tool_registry.py` (6 tools) |
| Run-level summary (total, by category, issues & assumptions) | `final_result` payload + `reports` row |
| Per-invoice structured output | `invoice_result` payload + `receipts` row |
| Trace: step boundaries, tool calls, planner decisions, errors, timing | `traces` table (primary) + Langfuse + JSON logs |
| README.md (setup, curl, SSE example, mock instructions) | `README.md` |
| spec.md (endpoint contract, event schema, tool schemas, output schema) | `spec.md` — **note**: repo currently contains `specs.md` (plural, user's original input that seeded this design). During implementation, `specs.md` is replaced by `spec.md` matching the PDF deliverable. |
| DESIGN.md (max 1 page) | `DESIGN.md` |
| /transcripts/ (AI coding tool interaction logs) | `transcripts/` |
| Sample run trace | `transcripts/sample-run-trace.json` |
| AGENTS.md / tool authorization config | `AGENTS.md` |

## 11. Non-goals for this build

- Multi-run concurrency / queue (single-process, one run at a time).
- Auth on `/runs/stream`.
- PDF / HEIC input (jpg/jpeg/png/webp only).
- Currency conversion (USD assumed when absent; flagged).
- Dashboard / UI.
- Horizontal scaling.
- Retry beyond one attempt on network errors.
- SSE reconnection / resume-from-seq.
- OCR / LLM cost optimization.
- Parallel per-invoice processing (per Q4).

## 12. Known risks

1. **Supabase cloud dependency** (Q5): reviewer must provision a Supabase project. Setup instructions must make this as smooth as possible; keep migrations runnable in one command.
2. **Timebox vs stack**: full hexagonal + Supabase + Alembic + Langfuse + LangGraph + two LLM providers is heavy for 2–3h. If slippage, slip in this order: Layer 3 repo tests → e2e smoke → Langfuse spans → Alembic refinements (keep the one revision). Never slip: event contract, SSE endpoint, mock mode, domain tests.
3. **Sub-agent structured output fragility**: DeepSeek must return strict JSON matching `Categorization`. Use provider response_format (JSON mode) or a schema-prompted system message with Pydantic validation. Schema mismatches are classified as Band A (invoice-level recoverable).
4. **Trace writer latency**: synchronous DB inserts inside the emit path could slow the stream. Acceptable at expected scale (single-digit invoices per run, short streams). If this becomes a bottleneck in practice, convert `TraceRepository` to an async queued writer behind the same port. Listed as a future improvement in `DESIGN.md`.
5. **Mock/real drift**: fake adapters' contracts must match real adapters' contracts. Mitigation: shared Pydantic schemas for adapter I/O; the e2e smoke test (Layer 4) runs manually once before submission.
