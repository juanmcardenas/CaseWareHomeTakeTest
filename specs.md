# Receipt Processing Agent — System Spec

## 1. Purpose

Design and build a local backend service that processes receipt image files through an AI-assisted workflow and exposes the execution through an HTTP API with Server-Sent Events (SSE).

The system must:

- Accept receipt inputs from a local folder path or uploaded files
- Extract relevant receipt data using OCR (OpenAI API)
- Normalize and categorize receipts into approved expense categories
- Aggregate totals by category and overall spend
- Stream execution progress, tool activity, and final output
- Persist run data and execution artifacts for review and observability

> **Terminology note:** The take-home PDF uses the word "invoice". This system normalizes on **"receipt"** throughout code, schemas, SSE events, and docs for internal consistency. A receipt here is any invoice-like expense record (invoice, till receipt, bill). The SSE event name `receipt_result` replaces the PDF's `invoice_result`; this is a deliberate deviation from the PDF streaming contract, called out in `DESIGN.md`.

---

## 2. Architecture

Hexagonal Architecture (Ports and Adapters):

- Domain: business rules and entities
- Application: orchestration, tool registry, categorization sub-agent
- Infrastructure: frameworks, DB, AI providers, HTTP

Principles:
- Dependencies point inward
- AI is an adapter, not business logic
- Tool registry is a strict boundary — the agent may only access receipt data via registered tools
- Outer workflow is deterministic (LangGraph state machine); bounded LLM non-determinism lives inside a single sub-agent call per receipt
- Wiring centralized in `composition_root.py`

---

## 3. Tech Stack

- Python 3.11+
- FastAPI (HTTP + SSE)
- LangChain + LangGraph (deterministic outer orchestration; sub-agent inside `categorize_receipt`)
- Langfuse (secondary tracing; DB `traces` is the primary trace)
- Supabase Cloud Postgres (accessed via SQLAlchemy)
- SQLAlchemy + Alembic
- Pydantic (strict domain + adapter I/O)
- DeepSeek-chat (categorization sub-agent; one call per receipt)
- OpenAI API (OCR)

---

## 4. Core Workflow

Deterministic outer pipeline, sequential per receipt, with a bounded LLM sub-agent on step 3c.

1. Start run; insert `reports` row; open SSE stream.
2. Load images from input (multipart upload or folder path).
3. For each receipt (sequential):
   a. Extract raw fields via OCR tool (OpenAI).
   b. Normalize (domain: dates, currencies, totals).
   c. Categorize via DeepSeek sub-agent → `{category, confidence, notes, issues[]}` with the user's optional prompt injected into its system message.
   d. Emit `receipt_result` and insert `receipts` row.
4. Aggregate totals (pure domain function).
5. Generate final report (deterministic assembly).
6. Emit `final_result`; update `reports` row; close stream.

---

## 5. API

### POST /runs/stream

- Starts processing
- Returns SSE stream

**Input (one of):**
- `multipart/form-data` with one or more receipt image files and an optional `prompt` field
- `application/json` with `{folder_path: string, prompt?: string}`

**Response:** `text/event-stream`.

**Allowed file extensions:** jpg, jpeg, png, webp.
**Limits:** max 10 MB per file, max 25 files per run. Configurable via env.

---

## 6. SSE Events

Event types, each carrying `run_id`, monotonic `seq`, ISO-8601 `ts`, `event_type`:

- `run_started`
- `progress` — step boundaries (`load_images`, `ocr`, `normalize`, `categorize`, `aggregate`, `generate_report`)
- `tool_call`
- `tool_result` — carries a bounded `result_summary`; full payload lives in DB
- `receipt_result` — terminal event per receipt; on error, `status="error"` with partial fields and issues
- `final_result` — run-level summary
- `error` — unrecoverable run-level failure; terminal

Receipt-scoped events additionally carry `receipt_id` (UUID).

---

## 7. Tool Registry

Six tools. The agent may only touch receipt data through these.

1. `load_images(input)` → `[ImageRef]`
2. `extract_receipt_fields(image_ref)` → `RawReceipt` (OpenAI OCR)
3. `normalize_receipt(raw)` → `NormalizedReceipt` (pure domain)
4. `categorize_receipt(normalized, prompt, allowed_categories)` → `Categorization` (DeepSeek sub-agent)
5. `aggregate(receipts)` → `Aggregates` (pure domain)
6. `generate_report(aggregates, receipts, issues)` → `Report` (deterministic)

Constraints:
- Each tool call is traced: emitted to SSE, persisted to `traces`, opened as a Langfuse span.
- Tool I/O is validated against Pydantic schemas at the boundary.

---

## 8. Domain Concepts

Entities:
- **Receipt** — normalized receipt data (vendor, receipt_date, receipt_number, total, currency, category, confidence, notes, issues)
- **Categorization** — `{category, confidence, notes, issues[]}`
- **Issue** — `{severity, code, message, receipt_id?}`
- **Aggregates** — `{total_spend, by_category}`
- **Report** — final run-level output

Allowed Categories (domain enum):
- Travel
- Meals & Entertainment
- Software / Subscriptions
- Professional Services
- Office Supplies
- Shipping / Postage
- Utilities
- Other (requires explanatory note)

---

## 9. Agent Design

- **Outer pipeline** is a deterministic LangGraph state machine.
- **Sub-agent** is a single DeepSeek call inside `categorize_receipt`. Input: normalized fields + allowed categories + user prompt (injected into system message). Output: strict Pydantic `Categorization`.
- **One LLM call per receipt.** No tool-use loop, no ReAct trajectory.
- **Prompt steering:** user prompt is free-form text, passed through to the sub-agent's system message and persisted on `reports.prompt`.
- **Where agency lives:** only at categorization + issue-flagging. Everywhere else is deterministic.

---

## 10. Persistence

Supabase Cloud Postgres via SQLAlchemy + Alembic (single initial revision).

Core tables:
- `reports` — one row per run (status, prompt, input_kind, total_spend, by_category JSONB, issues JSONB, error)
- `receipts` — one row per processed receipt (vendor, receipt_date, receipt_number, total, currency, category, confidence, notes, issues JSONB, raw_ocr JSONB, normalized JSONB, status, error)
- `traces` — one row per SSE event (`receipt_id` is a correlation UUID, not a FK; trace rows are written before their receipts row)

Write-through semantics:
- `reports` row inserted before the SSE stream opens.
- Every SSE event triggers a `traces` insert inside the emit path.
- `receipts` row inserted at `receipt_result`.
- `reports` row updated at `final_result` (or `error`).
- Replay a run via `SELECT payload FROM traces WHERE report_id = ? ORDER BY seq`.

Store per-receipt:
- raw OCR payload (full)
- normalized payload (full)
- category, confidence, notes, issues
- final output is assembled deterministically from receipts + aggregates

---

## 11. Observability

- **Primary:** `traces` table (write-through on every SSE event).
- **Secondary:** Langfuse spans, opened via a single `@traced_tool` decorator on every tool and the sub-agent.
- **Tertiary:** JSON logs to stdout.

Each trace captures:
- Step boundaries
- Tool calls with bounded inputs/outputs
- Sub-agent decisions (planner output summarized)
- Errors and retries
- Timing (`duration_ms` on `tool_result`)

---

## 12. Non-Functional Requirements

- Local-first execution (single process, no horizontal scaling)
- High observability and traceability
- Deterministic outer workflow; bounded LLM non-determinism inside the sub-agent only
- Replaceable AI providers behind adapter ports
- Mock mode (`LLM_MODE=mock`) swaps both OCR and LLM adapters for deterministic fakes — removes API-key friction for reviewers and makes tests deterministic
- Clear separation of concerns via hexagonal architecture

---

## 13. Configuration

Environment-based configuration. Loaded via Pydantic `Settings`.

- `LLM_MODE` — `mock` | `real`
- `SUPABASE_DB_URL` — Postgres connection string
- `OPENAI_API_KEY`, `DEEPSEEK_API_KEY` — required when `LLM_MODE=real`
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
- `ASSETS_DIR`
- Timeouts: `OCR_TIMEOUT_S=30`, `LLM_TIMEOUT_S=20`, `TOOL_WALL_TIMEOUT_S=45`
- Upload limits: `MAX_FILE_SIZE_MB=10`, `MAX_FILES_PER_RUN=25`, `ALLOWED_EXTENSIONS=jpg,jpeg,png,webp`

Missing required keys in `real` mode fail fast at startup.

---

## 14. Error Handling

Three severity bands:

- **A. Receipt-level recoverable** — OCR/LLM timeout, parse failure, invalid category from LLM, unreadable file. One retry on network-class errors (timeout/429/5xx). Skip downstream steps for that receipt; still emit `receipt_result` with `status="error"`; exclude from aggregation; run continues.
- **B. Receipt-level soft warnings** — missing receipt number, ambiguous currency, low OCR or sub-agent confidence, totals mismatch. Sub-agent emits these in `issues[]`; receipt is counted in aggregation.
- **C. Run-level unrecoverable** — DB unreachable at startup, no images, invalid request, 100% of receipts failed. Terminal `error` event; `reports.status='failed'`; stream closes.

Boundary rules:
- `EventBus.publish` never raises.
- Trace-writer failures never kill the stream.
- All tool wrappers share a single `@traced_tool` decorator for span + timing + try/except + event emission.
- Sub-agent output is validated against a strict Pydantic model; schema mismatch is classified Band A.

The final report's `issues_and_assumptions` section includes all receipt-level issues plus fixed run-level assumptions (allowed extensions, USD-default assumption, OCR-failed exclusion).

---

## 15. Testing Strategy

Four layers, prioritized by value given the 2–3h timebox:

1. **Domain tests** (pure logic) — aggregation math, normalization, Pydantic validators.
2. **Application tests** (workflow) — LangGraph happy path, receipt-level error, run-level error, sub-agent prompt steering, tool registry contract.
3. **Infrastructure tests** (API + DB) — `/runs/stream` contract test, write-through trace verification, repository round-trip (Supabase-reachable; skippable).
4. **End-to-end** (real OCR + real LLM) — marked `@pytest.mark.e2e`, skipped by default.

Slip order if timebox bites: Layer 3 repository tests → e2e → Langfuse spans → Alembic polish. Layer 1 never slips.

---

## 16. Security Considerations

- Validate file types and size at the HTTP boundary.
- Restrict filesystem access: `folder_path` input must resolve inside `ASSETS_DIR`.
- Secure API keys via environment variables.
- No arbitrary tool execution — tools are statically registered.
- No auth on `/runs/stream` (local-execution non-goal).

---

## 17. Deployment Model

- Local execution
- Single Python process (`uvicorn`)
- Supabase Cloud Postgres (reviewer needs a Supabase project + connection string — called out in `README.md`)

---

## 18. Assets Handling

- Sample receipt images stored in `./assets` (synthetic / public-domain).
- Service can access assets directly.
- Folder path input defaults to `ASSETS_DIR`.
- Uploaded files stored in a per-run temp directory for the duration of the run.

---

## 19. Design Summary

A local, observable AI agent system:
- FastAPI + SSE streaming endpoint (`POST /runs/stream`)
- LangGraph deterministic outer orchestration
- DeepSeek sub-agent bounded to categorization + issue flagging (one call per receipt)
- OpenAI OCR adapter (mockable)
- Strict tool-based execution through a 6-tool registry
- Hexagonal architecture with centralized wiring in `composition_root.py`
- Write-through persistence across `reports` / `receipts` / `traces`
- Full traceability via DB (primary) + Langfuse (secondary) + JSON logs

---

## 20. Deliberate PDF Deviations

- **SSE event name** `invoice_result` → `receipt_result` (for entity-naming consistency). All other required event names are preserved.
- **Terminology** "invoice" → "receipt" throughout code, schemas, and docs.

Both deviations are documented in `DESIGN.md`.
