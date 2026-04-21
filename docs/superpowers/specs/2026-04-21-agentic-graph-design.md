# Agentic Graph — Design

**Date:** 2026-04-21
**Status:** Approved by user; ready for implementation planning.
**Supersedes (for `src/application/graph.py` only):** portions of `2026-04-20-receipt-processing-agent-design.md` describing the deterministic state machine. All non-graph elements of that design (domain models, ports, event bus, tools, adapters, HTTP/SSE contract, DB schema, tracing) remain in force.

## 1. Problem & motivation

`src/application/graph.py` currently uses LangGraph only as a typed sequencer: `start → process_receipt (looped) → finalize`. The LLM is confined to one tool inside one node (`categorize_receipt`). This design replaces the deterministic body of each node with **per-node ReAct agents** built via `langchain.agents.create_agent`, while preserving determinism at the *tool* layer.

**Primary goal (d):** showcase an agentic architecture using LangGraph for the take-home review. Two proof-of-capability outcomes:

1. **Error recovery** — agent in `per_receipt_node` can observe a low-confidence OCR result and call `re_extract_with_hint`, or call `skip_receipt` on unrecoverable input.
2. **Prompt-driven flexibility** — agent in `ingest_node` can use `filter_by_prompt` to exclude images whose filenames do not match the user's free-text prompt.

**Non-goals:**
- Parallel receipt processing.
- Global-scope triage across all receipts by a single agent.
- Changes to adapters, HTTP layer, DB schema, SSE wire contract, or event types.
- Checkpointing / resumable runs.

## 2. Architecture

### 2.1 Graph shape — three nodes, deterministic edges

```
START → ingest_node → (cond) ─┬─▶ END (run-level error: no_images / all_images_filtered_out)
                              └─▶ per_receipt_node ⇄ (cond loop) ⇄ finalize_node → END
```

- `ingest_node` runs once.
- `per_receipt_node` runs **once per image** via a graph-level counter (`state.current`).
- `finalize_node` runs once.

Transitions between nodes are **deterministic** conditional edges evaluated on typed `RunState`. Agents decide tool use *within* their own node; routing between nodes is not agent-controlled.

### 2.2 Agent framework

Each node is a compiled `create_agent(...)` subgraph wrapped in a thin Python adapter (a "node wrapper") that projects typed `RunState` in, and typed `RunState` out.

```python
from langchain.agents import create_agent

ingest_subgraph = create_agent(
    model=chat_model,
    tools=[load_images_tool, filter_by_prompt_tool],
    system_prompt=INGEST_SYSTEM_PROMPT,
    state_schema=AgentState,
    max_iterations=8,
)
```

`create_react_agent` from `langgraph.prebuilt` is deprecated per the LangChain v1 migration guide and is **not** used.

### 2.3 Models

| Where | Model | Adapter |
|---|---|---|
| Node agents (ReAct loops) | **DeepSeek-chat** | New — `ChatModelPort` → `ChatOpenAI(base_url=deepseek_base_url, ...)` |
| `categorize_receipt` tool (internal) | **DeepSeek-chat** | Existing — `LLMPort.categorize` → `DeepSeekLLMAdapter` (unchanged) |
| `extract_receipt_fields` / `re_extract_with_hint` tools (internal) | **OpenAI gpt-4o-mini (vision)** | Existing — `OCRPort.extract` → `OpenAIOCRAdapter` (extended with optional `hint`) |
| All other tools | No LLM | Pure Python |

Node-level reasoning and tool-internal categorization are independent DeepSeek clients (different prompts; agent uses tool-calling, categorizer uses JSON-mode).

### 2.4 Preserved invariants

1. **All determinism lives in tools.** Agent reasoning is "pick a tool, observe, repeat" — never writes to the bus, DB, or state directly. Only tools touch I/O.
2. **SSE trace chronology.** Agents run serially; the per-receipt loop is graph-driven; no parallelism. Events on the bus remain strictly monotonic in `seq`.
3. **Per-receipt atomicity.** One receipt's failure does not stop the run.
4. **Bounded iteration.** Each node has a hard `max_iterations`. Exceeding it produces a typed error (receipt-level or run-level depending on the node).
5. **Wire contract unchanged.** No new SSE event types; no changes to existing event schemas; `seq` counter shared.

## 3. Components

### 3.1 Per-node agents

#### 3.1.1 `ingest_node`

- **Purpose:** resolve the input image set, applying any prompt-driven filtering.
- **System prompt (sketch):** "You ingest a batch of receipt images. Use `load_images` first. If the user supplied a prompt, use `filter_by_prompt` to exclude images that don't match. When you have the final list, stop."
- **Tools:** `load_images`, `filter_by_prompt`.
- **Input projection:** `{messages: [HumanMessage(content=user_prompt or "Process all receipts.")]}`.
- **Output projection:** wrapper writes `state.images = kept`, `state.filtered_out = dropped`.
- **`max_iterations`:** 8.

#### 3.1.2 `per_receipt_node`

- **Purpose:** fully process the receipt at `state.current`.
- **System prompt (sketch):** "You process **one** receipt (the one at `state.current`). Call `extract_receipt_fields`. If the raw output looks low-confidence or missing a total, call `re_extract_with_hint` once. Then call `normalize_receipt` and `categorize_receipt`. If any step fails unrecoverably, call `skip_receipt(reason)` and stop. Do not process other receipts."
- **Tools:** `extract_receipt_fields`, `re_extract_with_hint`, `normalize_receipt`, `categorize_receipt`, `skip_receipt`.
- **Input projection:** `{messages: [HumanMessage(f"Process receipt index {i+1}/{n}: source_ref={image.source_ref}")], image: state.images[i], receipt_id: uuid4()}`.
- **Output projection:** wrapper assembles a `Receipt` from the tools' results, calls `report_repo.insert_receipt(...)`, emits `ReceiptResult`, appends to `state.receipts`, increments `state.current`.
- **`max_iterations`:** 10.

#### 3.1.3 `finalize_node`

- **Purpose:** aggregate, narrate anomalies, produce the final report.
- **Pre-invocation short-circuit (deterministic):** if every receipt has `status="error"`, the wrapper emits `ErrorEvent(code="all_receipts_failed")` and does NOT invoke the agent. Preserves R4 from the original design.
- **System prompt (sketch):** "You produce the final report. Call `aggregate` first. Then call `detect_anomalies` on the aggregates + receipts; for each anomaly it returns, call `add_assumption` to record it in the issues list. Finally, call `generate_report`. Do not skip `generate_report`."
- **Tools:** `aggregate`, `detect_anomalies`, `add_assumption`, `generate_report`.
- **Input projection:** `{messages: [HumanMessage("Produce the final report.")], receipts: state.receipts, issues: state.issues}`.
- **Output projection:** `generate_report` tool emits `FinalResult` (same as today); wrapper returns final state.
- **`max_iterations`:** 12.

### 3.2 Node wrapper shape

Each node has a wrapper `async def _node(state: RunState) -> RunState`:

1. Build subgraph-input dict (projection-in).
2. `await subgraph.ainvoke(input, config={...})`.
3. Read tool-produced side effects from subgraph final state.
4. Return new `RunState` with typed fields updated.

Wrappers are the **only** place LangChain types leak into the application layer.

### 3.3 Tools

All tools are in `src/application/tool_registry.py`, decorated with `@traced_tool`, deterministic, and emit `tool_call`/`tool_result` events via the existing decorator.

#### 3.3.1 Existing tools (unchanged signatures)

- `load_images(ctx, *, loader) -> list[ImageRef]`
- `extract_receipt_fields(ctx, *, ocr, image) -> RawReceipt`  *(retries=1 on network errors)*
- `normalize_receipt(ctx, *, raw) -> NormalizedReceipt`
- `categorize_receipt(ctx, *, llm, normalized, user_prompt) -> Categorization`  *(retries=1 on network errors)*
- `aggregate_receipts(ctx, *, receipts) -> Aggregates`
- `generate_report(ctx, *, run_id, aggregates, receipts, issues) -> Report`  *(emits `FinalResult`)*

#### 3.3.2 New tools

| Tool | Signature | Purpose |
|---|---|---|
| `filter_by_prompt` | `(ctx, *, images: list[ImageRef], user_prompt: str \| None) -> FilterResult(kept: list[ImageRef], dropped: list[tuple[str, str]])` | Pure-Python keyword heuristic over `image.source_ref` (filename) and a hard-coded keyword map (e.g., "food"/"restaurant" → `["restaurant", "cafe", "food", "lunch", "dinner"]`). If `user_prompt` is None or contains no recognized keyword, returns all images as kept. **Intentionally coarse** — the point is showcasing that the agent *can* prune early via a tool; production-grade filtering is out of scope. **No LLM call** — avoids nested agents. |
| `re_extract_with_hint` | `(ctx, *, ocr: OCRPort, image: ImageRef, hint: str) -> RawReceipt` | Calls `OCRPort.extract(image, hint=...)`. Retries on network errors. |
| `skip_receipt` | `(ctx, *, receipt_id: UUID, reason: str) -> Receipt` | Returns a `Receipt(status="error", error=reason, issues=[Issue(severity="receipt_error", code="agent_skipped", message=reason, receipt_id=receipt_id)])`. Deterministic. |
| `detect_anomalies` | `(ctx, *, aggregates: Aggregates, receipts: list[Receipt]) -> list[Anomaly]` | Pure rules: single receipt ≥ 80% of spend, currency mismatch across receipts, ≥50% of receipts missing dates. |
| `add_assumption` | `(ctx, *, code: str, message: str) -> Issue` | Returns `Issue(severity="warning", code=code, message=message)`. The finalize wrapper collects every `add_assumption` tool-result from the subgraph's tool-messages and merges them into `state.assumptions_added_by_agent`, which the wrapper then unions with `state.issues` before passing to `generate_report`. No hidden process-global state. |

### 3.4 New port: `ChatModelPort`

```python
class ChatModelPort(ABC):
    @abstractmethod
    def build(self) -> BaseChatModel: ...
```

Two implementations:
- `DeepSeekChatModelAdapter(api_key, base_url, model, timeout_s)` → `ChatOpenAI(base_url=..., api_key=..., model=..., timeout=...)`.
- `FakeChatModelAdapter(scripted_messages)` → wraps `FakeMessagesListChatModel` for tests.

The port prevents `src/application/graph.py` from importing `langchain_openai` directly.

### 3.5 OCRPort extension

`OCRPort.extract` gains an optional `hint`:

```python
class OCRPort(ABC):
    @abstractmethod
    async def extract(self, image: ImageRef, hint: str | None = None) -> RawReceipt: ...
```

`OpenAIOCRAdapter` appends the `hint` (when present) to its system message. `MockOCRAdapter` ignores it (signature-compatible).

### 3.6 Domain additions

```python
class Anomaly(BaseModel):
    code: str
    message: str
    severity: Literal["warning", "notice"] = "warning"
```

### 3.7 `RunState` additions

```python
class RunState(BaseModel):
    images: list[ImageRef] = []
    filtered_out: list[tuple[str, str]] = []            # NEW
    receipts: list[Receipt] = []
    current: int = 0
    errors: list[str] = []
    issues: list[Issue] = []
    assumptions_added_by_agent: list[Issue] = []        # NEW
```

### 3.8 Composition root wiring

`src/composition_root.py` constructs a `ChatModelPort`:

```python
chat_model_port = (
    DeepSeekChatModelAdapter(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        timeout_s=settings.llm_timeout_s,
    )
    if settings.llm_mode == LLMMode.REAL
    else FakeChatModelAdapter(default_mock_script())
)
```

`default_mock_script()` is a shipped-with-repo script that makes a plausible run happen end-to-end without API keys — calls each tool in order, finishes — so mock mode still works.

### 3.9 Files touched

- **Rewritten:** `src/application/graph.py`.
- **Extended:** `src/application/tool_registry.py` (+5 tools), `src/application/ports.py` (+`ChatModelPort`, +`hint` param on `OCRPort.extract`), `src/composition_root.py`, `src/domain/models.py` (+`Anomaly`).
- **New:** `src/application/agent_prompts.py`, `src/infrastructure/llm/deepseek_chat_model.py`, `tests/fakes/fake_chat_model.py`.
- **Adapters extended:** `src/infrastructure/ocr/openai_adapter.py`, `src/infrastructure/ocr/mock_adapter.py` (optional `hint` param).
- **Deleted:** nothing.
- **New dependencies:** `langchain`, `langchain-openai`.

## 4. Data flow

### 4.1 End-to-end run (N images, user prompt)

1. **Entry.** SSE handler subscribes to the bus. Pre-node wrapper emits `RunStarted` before `ingest_node` runs.
2. **`ingest_node`:** wrapper projects `{messages: [HumanMessage(user_prompt or "Process all receipts.")]}`. Agent calls `load_images`; optionally calls `filter_by_prompt`. Wrapper writes `state.images`, `state.filtered_out`; emits `progress(step="ingest_done", n=len(kept))`.
3. **Conditional edge.** If `len(state.images) == 0 and len(state.filtered_out) == 0` → `ErrorEvent(code="no_images")` → END (nothing was found at all). If `len(state.images) == 0 and len(state.filtered_out) > 0` → `ErrorEvent(code="all_images_filtered_out")` → END (everything was filtered out by prompt). Else → `per_receipt_node`.
4. **`per_receipt_node`** (N times):
    - Wrapper projects input for receipt `state.current`.
    - Agent typical path: `extract_receipt_fields` → [`re_extract_with_hint`]? → `normalize_receipt` → `categorize_receipt` → finish.
    - Agent degraded path: `skip_receipt(reason)` → finish.
    - Wrapper assembles `Receipt`, persists via `report_repo.insert_receipt`, emits `ReceiptResult`, increments `state.current`.
5. **Loop edge.** `state.current < len(state.images)` → `per_receipt_node`. Else → `finalize_node`.
6. **`finalize_node`:**
    - If every receipt errored → `ErrorEvent(code="all_receipts_failed")` → END (agent not invoked).
    - Else: `aggregate` → `detect_anomalies` → `add_assumption` × K → `generate_report` (emits `FinalResult`).
7. **End.** SSE stream closes.

### 4.2 Event ordering (N=2, happy path)

```
run_started
progress(step="ingest_start")
tool_call(load_images) → tool_result(load_images)
[ tool_call(filter_by_prompt) → tool_result(filter_by_prompt) ]?
progress(step="ingest_done", n=2)
progress(step="process_receipt", i=1, n=2, receipt_id=…)
tool_call(extract_receipt_fields) → tool_result(extract_receipt_fields)
[ tool_call(re_extract_with_hint) → tool_result(re_extract_with_hint) ]?
tool_call(normalize_receipt) → tool_result(normalize_receipt)
tool_call(categorize_receipt) → tool_result(categorize_receipt)
receipt_result(receipt_id=…, status="ok", …)
progress(step="process_receipt", i=2, n=2, receipt_id=…)
… (same sequence) …
receipt_result(…)
progress(step="finalize_start")
tool_call(aggregate) → tool_result(aggregate)
tool_call(detect_anomalies) → tool_result(detect_anomalies)
[ tool_call(add_assumption) → tool_result(add_assumption) ] × K
tool_call(generate_report) → tool_result(generate_report)
final_result(…)
```

**Guarantees:** `seq` strictly monotonic. No new event types. Agent internal messages not emitted.

### 4.3 Mock mode

`FakeChatModelAdapter` with a shipped default script drives nodes deterministically. MockOCR/MockLLM/MockImageLoader remain as tool backends. End-to-end mock runs are reproducible byte-for-byte.

## 5. Error handling

### 5.1 Taxonomy

| Band | Definition | Producer | Surface |
|---|---|---|---|
| Tool-attempt | One tool call raised | `@traced_tool` | `tool_result(error=true, error_message=...)` on wire; agent sees it as observation |
| Receipt-level | One receipt can't complete OK | Agent (`skip_receipt`) or wrapper (`max_iterations`) | `receipt_result(status="error", ...)`; run continues |
| Run-level | Run can't produce `final_result` | Pre/post wrappers (deterministic checks) | `ErrorEvent(code, message)`; graph terminates |

### 5.2 Run-level error codes

| Code | Trigger | Location |
|---|---|---|
| `no_images` | `len(state.images) == 0` after ingest | Edge after ingest |
| `all_images_filtered_out` | Filter removed every image | Edge after ingest |
| `all_receipts_failed` | Every receipt `status="error"` | `finalize_node` wrapper |
| `ingest_iterations_exhausted` | Ingest agent hit `max_iterations` without populating images | `ingest_node` wrapper |
| `finalize_iterations_exhausted` | Finalize agent hit `max_iterations` without calling `generate_report` | `finalize_node` wrapper |
| `no_final_report` | Finalize agent finished without calling `generate_report` | `finalize_node` wrapper |

### 5.3 Retry policy

Two layers, both preserved / new:

1. **Network retry inside a tool** (existing). `@traced_tool(retries=1)` on network-sensitive tools. Transparent to the agent.
2. **Agent-level retry** (new). On any other tool failure, the agent observes `tool_result(error=true)` and can retry, try a different tool, or call `skip_receipt`. Bounded by `max_iterations`.

### 5.4 Failure behavior per node

- **`ingest_node`** iteration exhaustion → `ErrorEvent(code="ingest_iterations_exhausted")`.
- **`per_receipt_node`** iteration exhaustion OR agent finishes without producing a `Receipt` → wrapper synthesizes `Receipt(status="error", reason="receipt_iterations_exhausted" | "agent_did_not_finish")`; emits `ReceiptResult`; run continues.
- **`finalize_node`** iteration exhaustion → `finalize_iterations_exhausted`. Skipped `generate_report` → `no_final_report`. Tool raising inside finalize is observed by agent; if unrecoverable → iteration exhaustion path.

### 5.5 Malformed agent tool calls

`create_agent` validates tool inputs against the tool signature. Invalid calls are returned to the agent as structured validation errors and count toward `max_iterations`. Tools don't defensively validate.

### 5.6 Termination guarantees

- Every run ends with exactly one `final_result` OR `error`.
- Every processed receipt ends with exactly one `receipt_result`.
- `max_iterations` on each subgraph is the absolute bound; no unbounded agent loops.

## 6. Testing

### 6.1 Test pyramid

| Layer | Coverage | LLM |
|---|---|---|
| Unit — domain/tools | `normalize`, `aggregate`, `detect_anomalies` rules, `add_assumption` shape, `filter_by_prompt` heuristic | None |
| Unit — node wrappers | Projection-in / projection-out with canned subgraph output | None (subgraph mocked at `ainvoke`) |
| Unit — node subgraphs | Each compiled subgraph wired correctly: prompt → tool choice → observation → finish | `FakeMessagesListChatModel` with scripted AIMessages |
| Integration — full graph | Parent graph wiring; event ordering; state between nodes | Fake ChatModel with multi-node script |
| Contract — HTTP/SSE | `/runs/stream` produces same event types in same order | Mock model from composition root |
| Smoke — real API | End-to-end with DeepSeek + OpenAI; 1–2 images; loose invariants | Real models; `@pytest.mark.e2e`; credential-gated |

### 6.2 Fake ChatModel

`tests/fakes/fake_chat_model.py`:

```python
def fake_chat_model(script: list[AIMessage]) -> BaseChatModel:
    # wraps langchain_core FakeMessagesListChatModel
```

DSL helpers:

```python
def tool_call(tool_name: str, args: dict) -> AIMessage: ...
def finish(content: str = "") -> AIMessage: ...
```

### 6.3 Tests per node

**`ingest_node`:**
- Happy path: 3 images in → agent calls `load_images` → finishes → state has 3 images.
- Prompt-driven filter: user prompt "only restaurants" → script calls `load_images` then `filter_by_prompt` → state has 1 image.
- No prompt: `load_images` → finish (no filter).
- Iteration exhaustion: loops `load_images` → `max_iterations=8` → `ingest_iterations_exhausted`.

**`per_receipt_node`:**
- Happy OK: extract → normalize → categorize → finish.
- Re-extraction: low-confidence raw → `re_extract_with_hint` → normalize → categorize.
- Agent-driven skip: extract errors observed → `skip_receipt` called.
- Agent abandons without skip → wrapper synthesizes `Receipt(status="error", reason="agent_did_not_finish")`.
- Iteration exhaustion → `Receipt(status="error", reason="receipt_iterations_exhausted")`.

**`finalize_node`:**
- Happy path: aggregate → detect_anomalies (empty) → generate_report → finish.
- Anomaly path: detect_anomalies returns 2 anomalies → `add_assumption` × 2 → generate_report.
- R4: all receipts errored → agent not invoked → `ErrorEvent(code="all_receipts_failed")`.
- Missing `generate_report` → `ErrorEvent(code="no_final_report")`.

**Parent graph:**
- Full 2-receipt run: multi-segment script asserts event type sequence.
- Zero images: `ErrorEvent(code="no_images")`.
- All filtered out: `ErrorEvent(code="all_images_filtered_out")`.

### 6.4 Test file actions

| File | Action |
|---|---|
| `tests/application/test_graph.py` | Rewrite |
| `tests/application/test_event_bus.py` | Keep |
| `tests/application/test_events.py` | Keep |
| `tests/application/test_tool_registry.py` | Extend (+5 tools) |
| `tests/application/test_traced_tool.py` | Keep |
| `tests/application/test_subagent.py` | Keep |
| `tests/infrastructure/test_runs_stream.py` | Keep |
| `tests/infrastructure/test_repositories.py` | Keep |
| `tests/e2e/*` | Keep |

### 6.5 Determinism

- Non-e2e tests use scripted `FakeMessagesListChatModel`. Zero real-LLM calls.
- Existing MockOCR/MockLLM/MockImageLoader reused.
- Suite target < 5s, offline.

### 6.6 Explicitly out of scope for tests

- Quality of real model's tool-ordering choices (covered by e2e smoke + Langfuse review).
- Prompt wording (reviewed in PR; not asserted).
- LangChain internal message history (only external effects asserted).

## 7. Migration

- **M1 — replace `graph.py` outright.** Agentic becomes the only graph. No env flag, no parallel deterministic fallback.
- Existing deterministic tests in `tests/application/test_graph.py` rewritten against the fake ChatModel.
- Real-API e2e smoke test (`tests/e2e/`) runs against both old and new contracts since the wire contract is unchanged.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Agent ignores `generate_report` in finalize | Wrapper detects absence and emits `no_final_report`; explicit line in system prompt |
| Agent infinite-loops retrying a failing tool | `max_iterations` per node |
| LLM cost / latency bump vs. deterministic | Accepted (explicit goal is showcase); `max_iterations` bounds worst case |
| LangChain / LangGraph v1 API churn | Pin versions in `pyproject.toml`; `create_agent` is the stable recommended API |
| Mock mode regression (no real LLM to drive agents) | `FakeChatModelAdapter` with shipped `default_mock_script()` |
| SSE trace becomes noisy with agent reasoning | Agent internal messages deliberately NOT emitted to bus |

## 9. Open questions

None blocking. Two deferred choices (answered during brainstorm):
- Agent internal "thinking" messages on the wire → **no**, wire contract preserved.
- `filter_by_prompt` as LLM call vs. pure heuristic → **heuristic**, avoids nested agents.

## 10. Deliverables of the implementation plan (out of scope here)

- Phase-by-phase plan to be produced by the `writing-plans` skill after this spec is approved.
- Expected phases: ports & model adapter → new tools → agent prompts → node subgraphs → node wrappers → parent graph → tests rewrite → composition root → mock-script default.
