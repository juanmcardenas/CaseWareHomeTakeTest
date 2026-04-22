# Category-based `filter_by_prompt` — Design

**Date:** 2026-04-22
**Status:** Approved by user; ready for implementation planning.
**Supersedes:** filename-based keyword heuristic defined in `docs/superpowers/specs/2026-04-21-agentic-graph-design.md` §3.3.2 (`filter_by_prompt` tool). All other aspects of the agentic-graph design are unchanged.

## 1. Problem

The current `filter_by_prompt` tool is a pure-Python substring match over `ImageRef.source_ref` filenames (e.g. `restaurant_001.png` passes a "food" prompt; `receipt_001.jpg` does not). Real-world receipt images have opaque filenames, so in practice the filter never fires — a curl against `{"prompt": "focus on restaurants"}` against the project's 18 asset images dropped 0 receipts. The filter is effectively dead code.

This spec replaces the filename match with a filter that runs after receipts have been OCR'd, normalized, and categorized, so it can match on **category** instead of filename. Receipts that don't match the prompt are marked `status="filtered"` and naturally drop out of aggregation.

## 2. Decisions locked during brainstorm

| Decision | Value | Rationale |
|---|---|---|
| Pipeline placement | `finalize_node`, before `aggregate` | Per-receipt node stays simple; every receipt gets full metadata in the final report regardless of filter outcome |
| Matching surface | `Receipt.category` (the AllowedCategory enum) | Uses work already done by the categorizer; more reliable than keyword-matching against noisy OCR text |
| Matching strategy | Hand-curated keyword-to-category map | Deterministic, auditable, reuses the shape of the existing `_PROMPT_KEYWORD_MAP` |
| Directionality | Include + exclude | Detect negation words (`exclude`, `except`, `not`, `no `, `without`, `skip`); otherwise prompt is inclusive |
| Fate of filtered receipts | Mark `status="filtered"` in the final report, exclude from aggregates | Preserves the reviewable trace; matches the existing `ok`/`error` status convention |
| Old filename-based tool | Remove entirely; keep the name `filter_by_prompt` | Filename filtering is nearly useless and adds noise to the ingest agent's tool set |

## 3. Architecture

### 3.1 Pipeline change

Before:

```
ingest_node:        load_images → filter_by_prompt? → finish
per_receipt_node:   extract → [re_extract]? → normalize → categorize → finish
finalize_node:      aggregate → detect_anomalies → add_assumption* → generate_report
```

After:

```
ingest_node:        load_images → finish
per_receipt_node:   extract → [re_extract]? → normalize → categorize → finish    (unchanged)
finalize_node:      filter_by_prompt? → aggregate → detect_anomalies
                     → add_assumption* → generate_report
```

Tool registry: `filter_by_prompt` keeps its name but its signature changes and it moves from `ingest_node`'s tool list to `finalize_node`'s tool list.

### 3.2 Preserved invariants

- SSE wire contract: no new event types. `Receipt.status` gains one allowed value (`"filtered"`) alongside existing `"ok"`/`"error"`.
- Determinism at the tool layer. The filter tool is pure Python; no LLM call inside the filter.
- Per-receipt atomicity: filtering is a finalize-stage concern. Per-receipt processing is unchanged; every receipt emits exactly one `receipt_result` event with `status="ok"` or `status="error"` based on processing outcome. The `"filtered"` status only appears in `final_result.receipts`.
- R4 short-circuit (`all_receipts_failed`): runs *before* the filter, so if every receipt already failed, the filter never runs.

### 3.3 Intentional trade-off: `receipt_result` vs `final_result` status mismatch

A receipt's `receipt_result` SSE event is emitted by `per_receipt_node` with the processing outcome (`"ok"` or `"error"`). If that receipt is later filtered in finalize, its `status` in `final_result.receipts` will be `"filtered"` — different from the `"ok"` earlier on the wire.

This is accepted and documented. `receipt_result` answers "did we successfully process this receipt?"; `final_result.receipts[i].status` answers "what did this receipt contribute to the report?" Each event is truthful for its stage. No second `receipt_result` event is emitted post-filter.

## 4. Components

### 4.1 `_CATEGORY_KEYWORD_MAP` — `src/application/tool_registry.py`

```python
_CATEGORY_KEYWORD_MAP: dict[AllowedCategory, list[str]] = {
    AllowedCategory.MEALS: [
        "food", "meal", "meals", "restaurant", "cafe", "coffee",
        "lunch", "dinner", "breakfast", "dining", "entertainment",
    ],
    AllowedCategory.TRAVEL: [
        "travel", "flight", "airfare", "hotel", "airbnb", "lodging",
        "uber", "lyft", "taxi", "train", "transit", "transport", "transportation",
    ],
    AllowedCategory.SOFTWARE: [
        "software", "subscription", "saas", "app", "license",
    ],
    AllowedCategory.PROFESSIONAL: [
        "professional", "consulting", "consultant", "legal", "accounting", "advisory",
    ],
    AllowedCategory.OFFICE_SUPPLIES: [
        "office", "supplies", "stationery", "paper", "desk",
    ],
    AllowedCategory.SHIPPING: [
        "shipping", "postage", "mail", "delivery", "courier", "post",
    ],
    AllowedCategory.UTILITIES: [
        "utility", "utilities", "electric", "electricity", "water",
        "gas", "internet",
    ],
    # No entry for AllowedCategory.OTHER — users shouldn't have to include/exclude it by name.
}
```

### 4.2 `_parse_prompt` — pure helper in `tool_registry.py`

```python
_NEGATION_WORDS = ("exclude", "except", "not", "no ", "without", "skip")

def _parse_prompt(prompt: str) -> tuple[set[AllowedCategory], set[AllowedCategory]]:
    """
    Returns (include, exclude) category sets.
    - No recognized keyword → both sets empty; caller treats as "no filter".
    - Any negation word in prompt → matched categories go to 'exclude'; 'include' is empty.
    - Otherwise → matched categories go to 'include'; 'exclude' is empty.
    Negation is whole-prompt scope — simple and deterministic. "only food, no travel"
    is treated as an exclusion prompt because "no " matches.
    """
```

### 4.3 `filter_by_prompt` tool

```python
@traced_tool(
    "filter_by_prompt",
    summarize=lambda r: {
        "total": len(r),
        "kept": sum(1 for rc in r if rc.status == "ok"),
        "filtered": sum(1 for rc in r if rc.status == "filtered"),
    },
)
async def filter_by_prompt(
    ctx: ToolContext, *, receipts: list[Receipt], user_prompt: str | None,
) -> list[Receipt]:
    """Mark receipts that don't match the prompt as status='filtered'.

    No-op if prompt is empty or maps to no category. Receipts with status='error'
    are left untouched. Receipts whose status becomes 'filtered' gain an Issue
    (code='filtered_by_prompt', severity='warning') recording the reason.
    """
```

Decision logic per receipt:
- `r.status != "ok"` → unchanged.
- `include` set non-empty and `(r.category is None or r.category not in include)` → flip to `filtered`.
- `exclude` set non-empty and `r.category is not None and r.category in exclude` → flip to `filtered`.
- Otherwise → unchanged.

Flipping to `filtered` is done via `r.model_copy(update={"status": "filtered", "issues": r.issues + [new_issue]})`.

### 4.4 `Receipt.status` — `src/domain/models.py`

```python
status: Literal["ok", "error", "filtered"] = "ok"
```

Only literal change. `aggregate` already filters on `status == "ok"`, so filtered receipts drop from `total_spend` and `by_category` without touching aggregation code. `detect_anomalies` already scopes to `status == "ok"` — anomalies reflect the kept set.

### 4.5 Agent-facing builder — `src/infrastructure/agent_tools.py`

Old signature:

```python
def build_filter_by_prompt_tool(
    *, ctx_factory, images_provider: Callable[[], list[ImageRef]],
    user_prompt: str | None,
) -> StructuredTool: ...
```

New signature:

```python
def build_filter_by_prompt_tool(
    *, ctx_factory, receipts_provider: Callable[[], list[Receipt]],
    user_prompt: str | None,
) -> StructuredTool:
    async def _run() -> list[dict]:
        result = await filter_by_prompt(
            ctx_factory(), receipts=receipts_provider(), user_prompt=user_prompt,
        )
        return _dump(result)
    return StructuredTool.from_function(
        coroutine=_run, name="filter_by_prompt",
        description="Mark receipts that don't match the user prompt as status='filtered'. Takes no arguments.",
    )
```

Returns a list of receipt dicts (serialized via `_dump`). No agent-provided args.

### 4.6 Graph wiring — `src/application/graph.py`

**Ingest node:**
- Remove `filter_by_prompt` from tool list. New tool list: `[load_tool]` only.
- Remove `_capture_filter` helper entirely.
- `state.filtered_out` field on `RunState` becomes unused. Left in place for backward compatibility; emits an empty list.

**Finalize node:**
- Add `filter_by_prompt` to the tools list at position 0.
- Add `_capture_filtered_receipts` helper: replaces the mutable receipts holder used by subsequent tools (`aggregate`, `detect_anomalies`, `generate_report`) with the filter's output.
- `receipts_provider` closures in finalize change: they no longer close over `state.receipts` directly — they read from a mutable holder `receipts_holder: dict` initialized from `state.receipts` and updated by the filter's capture wrapper.

### 4.7 Agent prompts — `src/application/agent_prompts.py`

**Ingest (replace):**
```
You are the ingest agent. Your job is to load the input images.

1. Call `load_images` exactly once to retrieve the candidate images.
2. Stop. Do not call any tool more than necessary.
```

**Finalize (add step before aggregate):**
```
You are the finalize agent. Produce the final report.

Required sequence:
1. If the user prompt implies category filtering (e.g. "only food",
   "exclude travel"), call `filter_by_prompt` first. Otherwise skip to step 2.
2. Call `aggregate` on the (possibly filtered) receipts.
3. Call `detect_anomalies` on the aggregates and receipts.
4. For EACH anomaly, call `add_assumption` once.
5. Call `generate_report`. This is REQUIRED — do not skip it.
6. Stop.
```

### 4.8 Files touched summary

| File | Change |
|---|---|
| `src/application/tool_registry.py` | New `filter_by_prompt` signature + body; `_CATEGORY_KEYWORD_MAP` + `_parse_prompt`; delete `FilterResult` |
| `src/infrastructure/agent_tools.py` | `build_filter_by_prompt_tool` signature change |
| `src/application/graph.py` | Remove filter wiring from ingest; add to finalize; delete `_capture_filter`; add `_capture_filtered_receipts`; convert finalize receipts_provider to holder-based |
| `src/application/agent_prompts.py` | Update INGEST and FINALIZE prompts |
| `src/domain/models.py` | `Receipt.status` Literal gains `"filtered"` |
| `tests/application/test_tool_registry.py` | Rewrite filter tests (~11 total); delete `FilterResult` references |
| `tests/application/test_graph.py` | Update ingest filter test; add finalize filter tests; add full-graph test |

## 5. Data flow

### 5.1 Happy-path sequence (3 receipts, prompt `"only food"`, 2 MEALS + 1 TRAVEL)

1. `run_started` emitted.
2. `ingest_node`: agent calls `load_images` → `state.images` has 3 entries → finish.
3. `per_receipt_node` × 3: each receipt OCR'd, normalized, categorized. All three emit `receipt_result(status="ok", category=...)`.
4. `finalize_node`:
   - R4 check passes (not all errored).
   - `progress(finalize_start)`.
   - Agent calls `filter_by_prompt` → returns new list where 1 TRAVEL receipt has `status="filtered"`. Filter tool emits `tool_call`/`tool_result` with `summarize={"total":3,"kept":2,"filtered":1}`. Wrapper updates `receipts_holder` to the filtered list.
   - Agent calls `aggregate` → receives filtered list via holder → `total_spend` reflects only 2 MEALS.
   - Agent calls `detect_anomalies` → scoped to `status="ok"` receipts → uses filtered list.
   - Agent calls `generate_report` → emits `final_result` with all 3 receipts (2 `status="ok"`, 1 `status="filtered"`).

### 5.2 Edge cases

| Case | Behavior |
|---|---|
| Prompt is `None` or empty | Filter tool returns input unchanged |
| Prompt has no recognized category keyword (e.g. `"random text"`) | No-op |
| Prompt is `"exclude"` alone (negation word but no category) | No-op (matched set empty) |
| Prompt excludes every existing category | All OK receipts flip to `filtered`; aggregate sees empty → `total_spend=0` |
| Receipt has `status="error"` | Filter leaves it unchanged |
| Receipt has `status="ok"` but `category=None` | Filtered out when include-mode is active; kept when exclude-mode is active. This is an edge case: `status="ok"` receipts should always have a category since they passed categorize_receipt; None only appears if a future change breaks that invariant. |
| Receipt has `status="ok"` with `category=OTHER` | Treated like any other category. `"only food"` filters it out; `"exclude travel"` keeps it |
| Agent skips `filter_by_prompt` when it should have called it | No error — aggregates count all OK receipts (degraded behavior, not a bug) |

## 6. Error handling

No new run-level `ErrorEvent` codes.

One new `Issue` code on per-receipt issues: `filtered_by_prompt` (severity `warning`). Added to `Receipt.issues` when the filter flips a receipt to `status="filtered"`:

```python
Issue(
    severity="warning",
    code="filtered_by_prompt",
    message=f"filtered out by prompt '{user_prompt}' (category={r.category.value if r.category else 'None'})",
    receipt_id=r.id,
)
```

## 7. Testing

### 7.1 Filter-tool unit tests — `tests/application/test_tool_registry.py`

Rewrite the existing `filter_by_prompt` tests (currently 3 filename-based) with:

- `test_filter_by_prompt_no_prompt_is_noop`
- `test_filter_by_prompt_unknown_keyword_is_noop`
- `test_filter_by_prompt_include_keeps_matching_category` (prompt `"only food"`; mixed MEALS/TRAVEL)
- `test_filter_by_prompt_exclude_drops_matching_category` (prompt `"exclude travel"`)
- `test_filter_by_prompt_leaves_errored_receipts_alone`
- `test_filter_by_prompt_multi_category_include` (prompt `"only food and office"`)
- `test_filter_by_prompt_filtered_receipts_have_filtered_by_prompt_issue`
- `test_filter_by_prompt_returns_empty_ok_when_everything_excluded`

### 7.2 `_parse_prompt` unit tests

- `test_parse_prompt_detects_exclusion_on_each_negation_word` — parametric over `exclude/except/not/no /without/skip`.
- `test_parse_prompt_include_is_default_when_no_negation`.
- `test_parse_prompt_multiple_categories` (`"food and office"` → both in include set).

### 7.3 Graph tests — `tests/application/test_graph.py`

- **Update** `test_ingest_node_with_prompt_filter_drops_non_matching` → rename to `test_ingest_node_ignores_prompt_filtering_intent`. Keep fixture (two images, prompt `"only food"`); script now only calls `load_images` + `finish()` (no `filter_by_prompt` call because it isn't in the tool list anymore); assert both images survive in `state.images`.
- **New** `test_finalize_node_filter_excludes_non_matching_from_aggregate` — 2 ok receipts (MEALS + TRAVEL) + prompt `"only food"` + script calling `filter_by_prompt` → aggregates reflect only MEALS; `final_result.receipts` has one `status="filtered"`.
- **New** `test_finalize_node_no_filter_when_agent_skips_it` — same receipts, script skips `filter_by_prompt` → both receipts stay `status="ok"`; aggregate includes both.

### 7.4 Integration (full graph) test

- `test_full_graph_with_filter_prompt_yields_partial_aggregates` — assert event sequence includes `tool_call(filter_by_prompt)` between `progress(finalize_start)` and `tool_call(aggregate)`.

### 7.5 Tests removed

- `FilterResult` class tests (the class is deleted).

### 7.6 Tests unchanged

- Per-receipt node tests (unchanged flow).
- Finalize R4 short-circuit test (filter doesn't run).
- SSE contract tests (contract unchanged; existing assertions hold).

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Category-keyword map misses a common prompt phrasing | Easy to extend; the map is a dict literal |
| Negation-word detection is coarse (`"only food, no travel"` is treated as exclusion) | Accepted; documented in §4.2 docstring |
| Agent skips `filter_by_prompt` even when prompt implies filtering | Same class of "agent advisory" behavior as the existing finalize steps; acceptable for a showcase; a future enforcement could be wrapper-level |
| Mock mode's `default_mock_script()` doesn't include a filter call | Accepted; mock mode is reviewer convenience, not feature-complete |

## 9. Out of scope

- LLM-based semantic matching of prompt → receipts.
- Filtering by receipt_date range, total amount, or vendor.
- Removing filtered receipts from `final_result.receipts` entirely (they stay with `status="filtered"`).
- Adding a second `receipt_result` SSE event after filtering.
- Emitting a new `filter_applied` event or similar.
- Interactive prompt disambiguation ("did you mean TRAVEL or MEALS?").
- Supporting custom category definitions beyond `AllowedCategory`.

## 10. Open questions

None — all resolved during brainstorm.
