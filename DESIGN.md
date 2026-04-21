# DESIGN

One-page architecture summary for reviewers.

## Architecture
Hexagonal (Ports & Adapters). Three layers:
- **Domain** — Pydantic entities + pure functions (normalization, aggregation). No I/O.
- **Application** — LangGraph deterministic state machine, 6-tool registry, categorization sub-agent, SSE events, in-process EventBus, abstract ports.
- **Infrastructure** — FastAPI HTTP + SSE, SQLAlchemy + Alembic against Supabase Cloud Postgres, Langfuse + JSON logs tracers, OpenAI OCR, DeepSeek categorization, mock adapters.

Wiring is centralized in `composition_root.py` (the only module importing both application and infrastructure).

## Framework choice
LangChain + LangGraph for orchestration. The graph is deterministic end-to-end — no ReAct loop. The agent's "judgment" is bounded to a single DeepSeek call inside `categorize_receipt` (one LLM call per receipt). This honors the PDF's "planner decisions" tracing requirement while keeping the outer workflow testable and replayable.

## Planner / Execution / State separation
- **Planner:** deterministic LangGraph + the per-receipt sub-agent. The graph decides what runs next; the sub-agent decides the category + flags issues.
- **Execution:** tools (6 of them) are the only interface to receipt data. Each tool is decorated with `@traced_tool` (span + timing + event emission + retry + error classification).
- **State:** `RunState` (Pydantic) carries `images`, `receipts`, `current` cursor, `errors`, `issues`. Persistence is write-through: every SSE event becomes a `traces` row; receipts and reports are written at their terminal events.

## Model selection
- **OCR** — OpenAI `gpt-4o-mini` (vision, JSON mode). Cheap, capable, widely available.
- **Categorization** — DeepSeek `deepseek-chat` via its OpenAI-compatible API. Strong at structured classification; cost-effective.
- Both mockable via a single `LLM_MODE=mock` env switch; tests run entirely offline.

## What I'd improve for production
- **Async trace writer** — currently synchronous in the emit path. Move to a bounded queue + background writer to decouple DB latency from stream cadence.
- **Per-receipt parallelism** — trivial lift (already behind the `process_receipt` node); skipped here for a cleaner single-run trajectory.
- **Retry expansion** — one retry on network errors today; production would layer in token-bucket rate-limit handling and a circuit breaker around each provider.
- **Reviewer-runnability** — requires a Supabase Cloud project. Dual-mode (local Postgres via docker-compose + Supabase Cloud optional) is a five-line change and cuts onboarding friction.
- **E2E coverage** — real-API smoke test is manual; wire it into CI with a nightly job gated on API keys.

## Deliberate PDF deviations
- **`invoice_result` → `receipt_result`** in the SSE event set. Matches our internal "receipt" entity name. All other required event names preserved.
- **"Invoice" → "Receipt"** terminology throughout. Documented in `spec.md` and `specs.md`.
