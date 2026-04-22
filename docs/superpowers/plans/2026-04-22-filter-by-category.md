# Category-based `filter_by_prompt` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the filename-substring `filter_by_prompt` tool with a category-based filter that runs in `finalize_node` after full OCR + normalize + categorize, marking non-matching receipts as `status="filtered"` and excluding them from aggregates.

**Architecture:** Filter moves from `ingest_node` to `finalize_node`. Tool signature changes from `(images, user_prompt) -> FilterResult` to `(receipts, user_prompt) -> list[Receipt]` where non-matching receipts have their `status` flipped. A new `_CATEGORY_KEYWORD_MAP` + `_parse_prompt` helper in `tool_registry.py` maps prompts to `AllowedCategory` enum values with include/exclude semantics driven by negation-word detection.

**Tech Stack:** Python 3.11, Pydantic v2, LangChain/LangGraph (unchanged), pytest-asyncio.

**Reference spec:** `docs/superpowers/specs/2026-04-22-filter-by-category-design.md`.

---

## File Structure

**Files modified:**
- `src/domain/models.py` — `Receipt.status` literal adds `"filtered"`.
- `src/application/tool_registry.py` — delete `FilterResult`, `_PROMPT_KEYWORD_MAP`, `_matched_keywords`; add `_CATEGORY_KEYWORD_MAP`, `_NEGATION_WORDS`, `_parse_prompt`; rewrite `filter_by_prompt`.
- `src/infrastructure/agent_tools.py` — `build_filter_by_prompt_tool` signature: `images_provider` → `receipts_provider`.
- `src/application/graph.py` — `ingest_node` drops filter wiring; `finalize_node` gains filter wiring with mutable `receipts_holder`; delete `_capture_filter`; add `_capture_filtered_receipts`.
- `src/application/agent_prompts.py` — simplify `INGEST_SYSTEM_PROMPT`; add filter step to `FINALIZE_SYSTEM_PROMPT`.
- `tests/application/test_tool_registry.py` — delete 3 old filter tests; add 3 `_parse_prompt` tests + 8 new `filter_by_prompt` tests.
- `tests/application/test_graph.py` — rename ingest filter test; add 2 finalize filter tests + 1 full-graph filter test.

**Files NOT touched:** `per_receipt_node` and its tests, SSE wire contract, DB schema, composition root, OCR/LLM/ChatModel adapters.

---

## Phase 1 — Domain and helpers

### Task 1.1: Add `"filtered"` to `Receipt.status` literal

**Files:**
- Modify: `src/domain/models.py:80`
- Test: `tests/domain/test_models.py` (append)

- [ ] **Step 1: Append failing test to `tests/domain/test_models.py`**

```python
def test_receipt_accepts_filtered_status():
    from uuid import uuid4
    from domain.models import Receipt
    r = Receipt(id=uuid4(), source_ref="x.png", status="filtered")
    assert r.status == "filtered"
```

- [ ] **Step 2: Run, expect failure**

```bash
PYTHONPATH=src .venv/bin/pytest tests/domain/test_models.py::test_receipt_accepts_filtered_status -v
```

Expected: `ValidationError: status: Input should be 'ok' or 'error'`.

- [ ] **Step 3: Update `src/domain/models.py`**

Find the `Receipt` class. Change:

```python
    status: Literal["ok", "error"] = "ok"
```

to:

```python
    status: Literal["ok", "error", "filtered"] = "ok"
```

- [ ] **Step 4: Run, expect PASS**

```bash
PYTHONPATH=src .venv/bin/pytest tests/domain/test_models.py -v
```

Expected: all tests pass (existing tests unchanged because `"filtered"` is a widening of the type).

- [ ] **Step 5: Commit**

```bash
git add src/domain/models.py tests/domain/test_models.py
git commit -m "feat(domain): Receipt.status accepts filtered literal"
```

### Task 1.2: Add `_CATEGORY_KEYWORD_MAP` + `_parse_prompt` helper

**Files:**
- Modify: `src/application/tool_registry.py` (append helpers near the top, after existing imports)
- Test: `tests/application/test_tool_registry.py` (append 3 unit tests)

- [ ] **Step 1: Append failing tests to `tests/application/test_tool_registry.py`**

Append at the end of the file:

```python
import pytest as _pytest_for_parse_prompt  # alias if pytest already imported above
from application.tool_registry import _parse_prompt
from domain.models import AllowedCategory


@_pytest_for_parse_prompt.mark.parametrize("prompt,expected_exclude", [
    ("exclude travel", {AllowedCategory.TRAVEL}),
    ("except food", {AllowedCategory.MEALS}),
    ("not office", {AllowedCategory.OFFICE_SUPPLIES}),
    ("no travel please", {AllowedCategory.TRAVEL}),
    ("without software", {AllowedCategory.SOFTWARE}),
    ("skip utilities", {AllowedCategory.UTILITIES}),
])
def test_parse_prompt_detects_exclusion_on_each_negation_word(prompt, expected_exclude):
    include, exclude = _parse_prompt(prompt)
    assert include == set()
    assert exclude == expected_exclude


def test_parse_prompt_include_is_default_when_no_negation():
    include, exclude = _parse_prompt("only food")
    assert exclude == set()
    assert include == {AllowedCategory.MEALS}


def test_parse_prompt_multiple_categories():
    include, exclude = _parse_prompt("food and office supplies please")
    assert exclude == set()
    assert include == {AllowedCategory.MEALS, AllowedCategory.OFFICE_SUPPLIES}
```

(The alias import is defensive — if `pytest` is already imported at top-of-file, just remove that first line and use the existing `pytest` name on the `@pytest.mark.parametrize` decorator.)

- [ ] **Step 2: Run, expect failure**

```bash
PYTHONPATH=src .venv/bin/pytest tests/application/test_tool_registry.py -v -k parse_prompt
```

Expected: `ImportError: cannot import name '_parse_prompt' from 'application.tool_registry'`.

- [ ] **Step 3: Append helpers to `src/application/tool_registry.py`**

The file currently has (around line 124) an old `_PROMPT_KEYWORD_MAP` and `_matched_keywords` used by the old filename-based filter. **Delete** that old block (it's superseded here) — specifically, remove:

```python
_PROMPT_KEYWORD_MAP: dict[str, list[str]] = {
    "food": ["restaurant", "cafe", ...],
    ...
}

class FilterResult(BaseModel):
    ...

def _matched_keywords(prompt: str) -> list[str]:
    ...
```

(`FilterResult` will also be deleted in Task 2.1 — the empty shell is fine to leave temporarily, but simpler to delete it now since nothing else yet uses it. Keep the `# 7. filter_by_prompt — pure-Python keyword heuristic` section header as a placeholder for the new block.)

Also delete the old `async def filter_by_prompt(...)` function body (it gets rewritten in Task 2.1). Leave a stub so imports don't break:

```python
async def filter_by_prompt(ctx: ToolContext, **kwargs):
    """Stub — rewritten in Task 2.1. Kept here so imports survive the transition."""
    raise NotImplementedError("filter_by_prompt is being rewritten; see Task 2.1")
```

Now append — right below those deletions — the new helpers:

```python
# Category-based prompt parsing for filter_by_prompt (post-categorize filtering)
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
}

_NEGATION_WORDS: tuple[str, ...] = ("exclude", "except", "not", "no ", "without", "skip")


def _parse_prompt(prompt: str) -> tuple[set[AllowedCategory], set[AllowedCategory]]:
    """Return (include, exclude) category sets.

    - No recognised keyword → both sets empty (caller treats as no-op).
    - Any negation word in prompt → matched categories go to 'exclude'.
    - Otherwise → matched categories go to 'include'.
    """
    text = prompt.lower()
    is_exclusion = any(neg in text for neg in _NEGATION_WORDS)
    matched: set[AllowedCategory] = set()
    for cat, keywords in _CATEGORY_KEYWORD_MAP.items():
        if any(kw in text for kw in keywords):
            matched.add(cat)
    if is_exclusion:
        return set(), matched
    return matched, set()
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
PYTHONPATH=src .venv/bin/pytest tests/application/test_tool_registry.py -v -k parse_prompt
```

Expected: all 8 new tests pass (6 parametric + 2 unit). Full-file smoke:

```bash
PYTHONPATH=src .venv/bin/pytest tests/application/test_tool_registry.py -v
```

The three existing `filter_by_prompt` tests (filename-based) will FAIL now — the old function body was replaced with a stub that raises `NotImplementedError`. That's expected; Task 2.1 rewrites them.

- [ ] **Step 5: Commit**

```bash
git add src/application/tool_registry.py tests/application/test_tool_registry.py
git commit -m "feat(tools): _parse_prompt + _CATEGORY_KEYWORD_MAP for category filtering"
```

---

## Phase 2 — Rewrite the tool + agent-facing builder

### Task 2.1: Rewrite `filter_by_prompt` + update `build_filter_by_prompt_tool`

**Files:**
- Modify: `src/application/tool_registry.py` — replace the stub with the real implementation; update existing tests.
- Modify: `src/infrastructure/agent_tools.py` — change `build_filter_by_prompt_tool` signature.
- Modify: `tests/application/test_tool_registry.py` — delete 3 old tests, add 8 new ones.

- [ ] **Step 1: Replace the 3 existing filter_by_prompt tests**

Open `tests/application/test_tool_registry.py`. Find the three tests named `test_filter_by_prompt_no_prompt_keeps_all`, `test_filter_by_prompt_unknown_keyword_keeps_all`, `test_filter_by_prompt_food_keyword_matches_restaurant_filename` (they're filename-based). **Delete** all three.

Append these 8 new tests (at the same location, under the 3 `_parse_prompt` tests added in Task 1.2):

```python
from uuid import uuid4 as _uuid4
from decimal import Decimal as _Decimal
from domain.models import Receipt as _Receipt, AllowedCategory as _AC, Issue as _Issue
from application.tool_registry import filter_by_prompt as _filter_by_prompt


def _ok_receipt(category: _AC, total: str = "10.00", source_ref: str = "x") -> _Receipt:
    return _Receipt(
        id=_uuid4(), source_ref=source_ref, status="ok",
        category=category, confidence=0.9, notes="n",
        total=_Decimal(total), currency="USD",
    )


def _error_receipt(source_ref: str = "y") -> _Receipt:
    return _Receipt(id=_uuid4(), source_ref=source_ref, status="error", error="boom")


@pytest.mark.asyncio
async def test_filter_by_prompt_no_prompt_is_noop():
    receipts = [_ok_receipt(_AC.MEALS), _ok_receipt(_AC.TRAVEL)]
    out = await _filter_by_prompt(_fctx(), receipts=receipts, user_prompt=None)
    assert all(r.status == "ok" for r in out)
    assert len(out) == 2


@pytest.mark.asyncio
async def test_filter_by_prompt_unknown_keyword_is_noop():
    receipts = [_ok_receipt(_AC.MEALS), _ok_receipt(_AC.TRAVEL)]
    out = await _filter_by_prompt(_fctx(), receipts=receipts, user_prompt="arbitrary freeform text")
    assert all(r.status == "ok" for r in out)


@pytest.mark.asyncio
async def test_filter_by_prompt_include_keeps_matching_category():
    meals = _ok_receipt(_AC.MEALS, source_ref="m")
    travel = _ok_receipt(_AC.TRAVEL, source_ref="t")
    out = await _filter_by_prompt(_fctx(), receipts=[meals, travel], user_prompt="only food")
    out_by_ref = {r.source_ref: r for r in out}
    assert out_by_ref["m"].status == "ok"
    assert out_by_ref["t"].status == "filtered"


@pytest.mark.asyncio
async def test_filter_by_prompt_exclude_drops_matching_category():
    meals = _ok_receipt(_AC.MEALS, source_ref="m")
    travel = _ok_receipt(_AC.TRAVEL, source_ref="t")
    out = await _filter_by_prompt(_fctx(), receipts=[meals, travel], user_prompt="exclude travel")
    out_by_ref = {r.source_ref: r for r in out}
    assert out_by_ref["m"].status == "ok"
    assert out_by_ref["t"].status == "filtered"


@pytest.mark.asyncio
async def test_filter_by_prompt_leaves_errored_receipts_alone():
    errored = _error_receipt(source_ref="e")
    ok = _ok_receipt(_AC.TRAVEL, source_ref="o")
    out = await _filter_by_prompt(_fctx(), receipts=[errored, ok], user_prompt="only food")
    out_by_ref = {r.source_ref: r for r in out}
    assert out_by_ref["e"].status == "error"   # untouched
    assert out_by_ref["o"].status == "filtered"


@pytest.mark.asyncio
async def test_filter_by_prompt_multi_category_include():
    meals = _ok_receipt(_AC.MEALS, source_ref="m")
    office = _ok_receipt(_AC.OFFICE_SUPPLIES, source_ref="o")
    travel = _ok_receipt(_AC.TRAVEL, source_ref="t")
    out = await _filter_by_prompt(
        _fctx(), receipts=[meals, office, travel],
        user_prompt="food and office supplies",
    )
    out_by_ref = {r.source_ref: r for r in out}
    assert out_by_ref["m"].status == "ok"
    assert out_by_ref["o"].status == "ok"
    assert out_by_ref["t"].status == "filtered"


@pytest.mark.asyncio
async def test_filter_by_prompt_filtered_receipts_have_issue():
    travel = _ok_receipt(_AC.TRAVEL, source_ref="t")
    out = await _filter_by_prompt(_fctx(), receipts=[travel], user_prompt="only food")
    assert out[0].status == "filtered"
    codes = [iss.code for iss in out[0].issues]
    assert "filtered_by_prompt" in codes
    filt_issue = next(iss for iss in out[0].issues if iss.code == "filtered_by_prompt")
    assert filt_issue.severity == "warning"
    assert "only food" in filt_issue.message
    assert "Travel" in filt_issue.message


@pytest.mark.asyncio
async def test_filter_by_prompt_returns_all_filtered_when_nothing_matches():
    meals = _ok_receipt(_AC.MEALS, source_ref="m")
    travel = _ok_receipt(_AC.TRAVEL, source_ref="t")
    out = await _filter_by_prompt(
        _fctx(), receipts=[meals, travel], user_prompt="only utilities",
    )
    assert all(r.status == "filtered" for r in out)
```

- [ ] **Step 2: Run, expect failure**

```bash
PYTHONPATH=src .venv/bin/pytest tests/application/test_tool_registry.py -v -k filter_by_prompt
```

Expected: 8 failures — the stub from Task 1.2 raises `NotImplementedError`.

- [ ] **Step 3: Rewrite the stub into the real `filter_by_prompt` in `src/application/tool_registry.py`**

Find the stub:

```python
async def filter_by_prompt(ctx: ToolContext, **kwargs):
    """Stub — ..."""
    raise NotImplementedError(...)
```

Replace with:

```python
# 7. filter_by_prompt — category-based post-categorization filter
def _summarize_filter_result(receipts: list[Receipt]) -> dict:
    return {
        "total": len(receipts),
        "kept": sum(1 for r in receipts if r.status == "ok"),
        "filtered": sum(1 for r in receipts if r.status == "filtered"),
    }


@traced_tool("filter_by_prompt", summarize=_summarize_filter_result)
async def filter_by_prompt(
    ctx: ToolContext, *, receipts: list[Receipt], user_prompt: str | None,
) -> list[Receipt]:
    """Mark receipts that don't match the prompt as status='filtered'.

    No-op if prompt is empty or maps to no category. Receipts with
    status='error' are left untouched. Filtered receipts gain an
    Issue(code='filtered_by_prompt', severity='warning').
    """
    if not user_prompt:
        return list(receipts)
    include, exclude = _parse_prompt(user_prompt)
    if not include and not exclude:
        return list(receipts)

    out: list[Receipt] = []
    for r in receipts:
        if r.status != "ok":
            out.append(r)
            continue
        cat = r.category
        flip = False
        if include and (cat is None or cat not in include):
            flip = True
        elif exclude and cat is not None and cat in exclude:
            flip = True

        if flip:
            filt_issue = Issue(
                severity="warning",
                code="filtered_by_prompt",
                message=(
                    f"filtered out by prompt {user_prompt!r} "
                    f"(category={cat.value if cat else 'None'})"
                ),
                receipt_id=r.id,
            )
            out.append(r.model_copy(update={
                "status": "filtered",
                "issues": r.issues + [filt_issue],
            }))
        else:
            out.append(r)
    return out
```

- [ ] **Step 4: Update `build_filter_by_prompt_tool` in `src/infrastructure/agent_tools.py`**

Find the current definition:

```python
def build_filter_by_prompt_tool(
    *, ctx_factory: Callable[[], ToolContext],
    images_provider: Callable[[], list[ImageRef]],
    user_prompt: str | None,
) -> StructuredTool:
    async def _run() -> dict:
        result = await filter_by_prompt(ctx_factory(), images=images_provider(), user_prompt=user_prompt)
        return _dump(result)

    return StructuredTool.from_function(
        coroutine=_run,
        name="filter_by_prompt",
        description="Filter the loaded images based on the user's prompt. Takes no arguments; uses images loaded by load_images and the run's user_prompt.",
    )
```

Replace with:

```python
def build_filter_by_prompt_tool(
    *, ctx_factory: Callable[[], ToolContext],
    receipts_provider: Callable[[], list[Receipt]],
    user_prompt: str | None,
) -> StructuredTool:
    async def _run() -> list[dict]:
        result = await filter_by_prompt(
            ctx_factory(), receipts=receipts_provider(), user_prompt=user_prompt,
        )
        return _dump(result)

    return StructuredTool.from_function(
        coroutine=_run,
        name="filter_by_prompt",
        description="Mark receipts that don't match the user's prompt as status='filtered'. Takes no arguments; uses the processed receipts and the run's user_prompt.",
    )
```

Also confirm the import at top of `agent_tools.py` includes `Receipt` — if the line currently reads:

```python
from domain.models import Issue, Receipt, Report
```

that covers it. If `Receipt` is missing from the imports, add it.

Also the old `ImageRef` import was used only by this builder's removed `images_provider` — check if `ImageRef` is used elsewhere in `agent_tools.py`. Most likely YES (by `build_extract_receipt_fields_tool` / `build_re_extract_with_hint_tool`). Leave the import in place.

- [ ] **Step 5: Run tool tests**

```bash
PYTHONPATH=src .venv/bin/pytest tests/application/test_tool_registry.py -v
```

Expected: all tests pass — the 3 old filter tests are gone, 8 new filter tests + 8 `_parse_prompt` tests pass, plus the existing non-filter tool tests.

- [ ] **Step 6: Commit**

```bash
git add src/application/tool_registry.py src/infrastructure/agent_tools.py tests/application/test_tool_registry.py
git commit -m "feat(tools): filter_by_prompt filters by category with include/exclude"
```

---

## Phase 3 — Rewire graph

### Task 3.1: Move filter from `ingest_node` to `finalize_node`

**Files:**
- Modify: `src/application/graph.py` — `ingest_node` drops filter wiring; `finalize_node` adds it with mutable holder; delete `_capture_filter`; add `_capture_filtered_receipts`.
- Modify: `tests/application/test_graph.py` — rename ingest filter test; add 2 finalize filter tests; add 1 full-graph filter test.

- [ ] **Step 1: Update `tests/application/test_graph.py` — rename ingest filter test**

Find the test `test_ingest_node_with_prompt_filter_drops_non_matching`. **Rename** it to `test_ingest_node_ignores_prompt_filtering_intent` and update its body:

```python
@pytest.mark.asyncio
async def test_ingest_node_ignores_prompt_filtering_intent():
    """ingest_node no longer has a filter tool — filtering moved to finalize.
    Even with a filter-shaped prompt, all loaded images pass through."""
    images = [_img("restaurant.png"), _img("uber.png")]
    script = [
        tool_call("load_images", {}),
        finish(),
    ]
    r = _runner(prompt="only food", images=images, script=script)
    state = await r.ingest_node(RunState())
    assert len(state.images) == 2
    # filtered_out remains empty — no filter runs in ingest anymore
    assert state.filtered_out == []
```

- [ ] **Step 2: Append 3 new graph tests to `tests/application/test_graph.py`**

Append AFTER `test_finalize_node_missing_generate_report_emits_no_final_report_error`:

```python
@pytest.mark.asyncio
async def test_finalize_node_filter_excludes_non_matching_from_aggregate():
    """When the finalize agent calls filter_by_prompt with a category-matching prompt,
    only matching receipts survive to aggregate."""
    ok_meals = Receipt(
        id=uuid4(), source_ref="m.png", status="ok",
        category=AllowedCategory.MEALS, confidence=0.9, notes="x",
        total=Decimal("25.00"), currency="USD",
    )
    ok_travel = Receipt(
        id=uuid4(), source_ref="t.png", status="ok",
        category=AllowedCategory.TRAVEL, confidence=0.9, notes="x",
        total=Decimal("100.00"), currency="USD",
    )
    bus = InMemoryEventBus()
    script = [
        tool_call("filter_by_prompt", {}),
        tool_call("aggregate", {}),
        tool_call("detect_anomalies", {}),
        tool_call("generate_report", {}),
        finish(),
    ]
    r = GraphRunner(
        run_id=uuid4(), prompt="only food",
        bus=bus, tracer=_NullTracer(),
        image_loader=MockImageLoader([]), ocr=MockOCR(), llm=MockLLM(),
        chat_model_port=FakeChatModelAdapter(script),
        report_repo=InMemoryReportRepository(),
    )
    state = RunState(receipts=[ok_meals, ok_travel])
    state = await r.finalize_node(state)

    final_events = [e for e in bus.published if e.get("event_type") == "final_result"]
    assert len(final_events) == 1
    final = final_events[0]
    # total_spend reflects only the MEALS receipt (25.00); TRAVEL is filtered
    assert final["total_spend"] == "25.00"
    # All receipts still appear in the report with their final status
    receipts_by_ref = {r["source_ref"]: r for r in final["receipts"]}
    assert receipts_by_ref["m.png"]["status"] == "ok"
    assert receipts_by_ref["t.png"]["status"] == "filtered"


@pytest.mark.asyncio
async def test_finalize_node_no_filter_when_agent_skips_it():
    """When the finalize agent doesn't call filter_by_prompt, all OK receipts
    count toward aggregates as normal — even if the prompt implied filtering."""
    ok_meals = Receipt(
        id=uuid4(), source_ref="m.png", status="ok",
        category=AllowedCategory.MEALS, confidence=0.9, notes="x",
        total=Decimal("25.00"), currency="USD",
    )
    ok_travel = Receipt(
        id=uuid4(), source_ref="t.png", status="ok",
        category=AllowedCategory.TRAVEL, confidence=0.9, notes="x",
        total=Decimal("100.00"), currency="USD",
    )
    bus = InMemoryEventBus()
    # Script skips filter_by_prompt
    script = [
        tool_call("aggregate", {}),
        tool_call("detect_anomalies", {}),
        tool_call("generate_report", {}),
        finish(),
    ]
    r = GraphRunner(
        run_id=uuid4(), prompt="only food",
        bus=bus, tracer=_NullTracer(),
        image_loader=MockImageLoader([]), ocr=MockOCR(), llm=MockLLM(),
        chat_model_port=FakeChatModelAdapter(script),
        report_repo=InMemoryReportRepository(),
    )
    state = RunState(receipts=[ok_meals, ok_travel])
    state = await r.finalize_node(state)

    final = [e for e in bus.published if e.get("event_type") == "final_result"][0]
    assert final["total_spend"] == "125.00"  # both counted
    receipts_by_ref = {r["source_ref"]: r for r in final["receipts"]}
    assert receipts_by_ref["m.png"]["status"] == "ok"
    assert receipts_by_ref["t.png"]["status"] == "ok"


@pytest.mark.asyncio
async def test_full_graph_with_filter_prompt_yields_partial_aggregates():
    """End-to-end: filter_by_prompt between finalize_start and aggregate in the event stream."""
    images = [_img("a.png"), _img("b.png")]
    ocr = MockOCR(responses={
        "a.png": RawReceipt(source_ref="a.png", vendor="Cafe", receipt_date="2024-03-01",
                            total_raw="$25.00", ocr_confidence=0.95),
        "b.png": RawReceipt(source_ref="b.png", vendor="Uber", receipt_date="2024-03-02",
                            total_raw="$100.00", ocr_confidence=0.95),
    })

    class _TwoCategoryLLM(MockLLM):
        async def categorize(self, normalized, allowed, user_prompt):
            from domain.models import Categorization, AllowedCategory
            if normalized.vendor == "Cafe":
                return Categorization(
                    category=AllowedCategory.MEALS, confidence=0.9, notes="cafe",
                )
            return Categorization(
                category=AllowedCategory.TRAVEL, confidence=0.9, notes="uber",
            )

    llm = _TwoCategoryLLM()
    script = [
        # ingest
        tool_call("load_images", {}), finish(),
        # per_receipt × 2
        tool_call("extract_receipt_fields", {}),
        tool_call("normalize_receipt", {}),
        tool_call("categorize_receipt", {}),
        finish(),
        tool_call("extract_receipt_fields", {}),
        tool_call("normalize_receipt", {}),
        tool_call("categorize_receipt", {}),
        finish(),
        # finalize with filter
        tool_call("filter_by_prompt", {}),
        tool_call("aggregate", {}),
        tool_call("detect_anomalies", {}),
        tool_call("generate_report", {}),
        finish(),
    ]
    bus = InMemoryEventBus()
    r = GraphRunner(
        run_id=uuid4(), prompt="only food",
        bus=bus, tracer=_NullTracer(),
        image_loader=MockImageLoader(images), ocr=ocr, llm=llm,
        chat_model_port=FakeChatModelAdapter(script),
        report_repo=InMemoryReportRepository(),
    )
    from application.graph import build_graph
    graph = build_graph(r)
    await graph.ainvoke(RunState())

    # Event ordering: finalize_start then filter_by_prompt then aggregate
    events = bus.published
    def _find_seq(evt_type: str, tool_name: str | None = None) -> int:
        for e in events:
            if e.get("event_type") == evt_type and (tool_name is None or e.get("tool") == tool_name):
                return e["seq"]
        raise AssertionError(f"no {evt_type}/{tool_name} event found")

    finalize_start_seq = None
    for e in events:
        if e.get("event_type") == "progress" and e.get("step") == "finalize_start":
            finalize_start_seq = e["seq"]
            break
    assert finalize_start_seq is not None
    filter_seq = _find_seq("tool_call", "filter_by_prompt")
    aggregate_seq = _find_seq("tool_call", "aggregate")
    assert finalize_start_seq < filter_seq < aggregate_seq

    # Final totals reflect only Cafe (MEALS, 25.00)
    final = [e for e in events if e.get("event_type") == "final_result"][0]
    assert final["total_spend"] == "25.00"
```

- [ ] **Step 3: Run the new tests, expect failures**

```bash
PYTHONPATH=src .venv/bin/pytest tests/application/test_graph.py -v
```

Expected: the 3 new/renamed tests fail — ingest still calls the old filter; finalize doesn't have `filter_by_prompt` in its tool list; the receipts_holder doesn't exist.

- [ ] **Step 4: Modify `src/application/graph.py` — `ingest_node`: drop filter wiring**

Find the `ingest_node` method. Locate this block:

```python
        # Holders for capturing tool outputs
        images_holder: list[ImageRef] = []
        dropped_holder: list[tuple[str, str]] = []

        # Construct the two tools
        load_tool = build_load_images_tool(
            ctx_factory=lambda: self._ctx(),
            loader=self.image_loader,
        )
        filter_tool = build_filter_by_prompt_tool(
            ctx_factory=lambda: self._ctx(),
            images_provider=lambda: list(images_holder),
            user_prompt=self.prompt,
        )

        # Wrap so their outputs are captured into the holders
        wrapped_load = _capture_list(load_tool, images_holder, ImageRef)
        wrapped_filter = _capture_filter(filter_tool, images_holder, dropped_holder)

        # Build the subgraph agent
        agent = create_agent(
            model=self.chat_model_port.build(),
            tools=[wrapped_load, wrapped_filter],
            system_prompt=INGEST_SYSTEM_PROMPT,
        )
```

Replace with:

```python
        # Holders for capturing tool outputs
        images_holder: list[ImageRef] = []

        # Construct the load_images tool (filter moved to finalize_node)
        load_tool = build_load_images_tool(
            ctx_factory=lambda: self._ctx(),
            loader=self.image_loader,
        )
        wrapped_load = _capture_list(load_tool, images_holder, ImageRef)

        # Build the subgraph agent — ingest only loads now
        agent = create_agent(
            model=self.chat_model_port.build(),
            tools=[wrapped_load],
            system_prompt=INGEST_SYSTEM_PROMPT,
        )
```

Then in the `return state.model_copy(...)` block at the end of `ingest_node`, drop the `filtered_out` update:

```python
        await self._progress("ingest_done", n=len(images_holder))
        return state.model_copy(update={
            "images": list(images_holder),
        })
```

Also **remove the import** of `build_filter_by_prompt_tool` from the top of `graph.py` IF it's no longer used — but we ARE about to use it in `finalize_node`, so keep the import.

Also **remove `_capture_filter`** from the module-level helper block (it's no longer used anywhere). Delete the entire function.

- [ ] **Step 5: Modify `src/application/graph.py` — add `_capture_filtered_receipts` helper at module scope**

Add (near `_capture_aggregates` or any other capture helper):

```python
def _capture_filtered_receipts(tool, receipts_holder: dict):
    """Wrap filter_by_prompt so its returned list of receipts is written into
    receipts_holder['receipts']. Later tools (aggregate / detect_anomalies /
    generate_report) read from this holder instead of state.receipts directly.
    """
    from langchain_core.tools import StructuredTool
    original_coro = tool.coroutine

    async def _wrapped(*args, **kwargs):
        result = await original_coro(*args, **kwargs)
        # result is a list[dict] from _dump(list[Receipt])
        hydrated: list[Receipt] = []
        if isinstance(result, list):
            for d in result:
                if isinstance(d, Receipt):
                    hydrated.append(d)
                elif isinstance(d, dict):
                    hydrated.append(Receipt(**d))
        receipts_holder["receipts"] = hydrated
        return result

    return StructuredTool.from_function(
        coroutine=_wrapped, name=tool.name, description=tool.description,
        args_schema=tool.args_schema,
    )
```

- [ ] **Step 6: Modify `src/application/graph.py` — `finalize_node`: add filter with holder-based receipts_provider**

Find `finalize_node`. Before the `aggregate_tool` construction, add:

```python
        # Mutable receipts holder: filter_by_prompt (if called) replaces this
        # with a filtered list. Aggregate/anomalies/report all read from here.
        receipts_holder: dict = {"receipts": list(state.receipts)}

        filter_tool = _capture_filtered_receipts(
            build_filter_by_prompt_tool(
                ctx_factory=lambda: self._ctx(),
                receipts_provider=lambda: list(receipts_holder["receipts"]),
                user_prompt=self.prompt,
            ),
            receipts_holder,
        )
```

Then **change every `receipts_provider=lambda: list(state.receipts)`** in finalize_node's tool construction to `receipts_provider=lambda: list(receipts_holder["receipts"])`. There are three such lines — in `build_aggregate_tool`, `build_detect_anomalies_tool`, and `build_generate_report_tool`. Example:

```python
        aggregate_tool = _capture_aggregates(
            build_aggregate_tool(
                ctx_factory=lambda: self._ctx(),
                receipts_provider=lambda: list(receipts_holder["receipts"]),
            ),
            aggregates_holder,
        )

        detect_tool = build_detect_anomalies_tool(
            ctx_factory=lambda: self._ctx(),
            aggregates_holder=aggregates_holder,
            receipts_provider=lambda: list(receipts_holder["receipts"]),
        )

        generate_report_tool = build_generate_report_tool(
            ctx_factory=lambda: self._ctx(),
            run_id=self.run_id,
            aggregates_holder=aggregates_holder,
            receipts_provider=lambda: list(receipts_holder["receipts"]),
            issues_provider=_issues_provider,
            report_holder=report_holder,
            emit_final_result=_emit_final,
        )
```

Finally, add the filter tool to the agent's tools list (at position 0, so the agent sees it first):

```python
        agent = create_agent(
            model=self.chat_model_port.build(),
            tools=[filter_tool, aggregate_tool, detect_tool, add_assumption_tool, generate_report_tool],
            system_prompt=FINALIZE_SYSTEM_PROMPT,
        )
```

- [ ] **Step 7: Run full graph tests, expect PASS**

```bash
PYTHONPATH=src .venv/bin/pytest tests/application/test_graph.py -v
```

Expected: all graph tests pass, including the 3 new/renamed ones.

If `test_finalize_node_happy_path_emits_final_result` (an existing test) now fails because its script doesn't include `filter_by_prompt`, that's a false failure: without a filter-shaped prompt, the agent is allowed to skip it. The test passes `prompt=None`, so the filter is a no-op regardless. If the script does call every remaining tool, the test should still pass.

Check any other existing finalize tests — if one of them now fails because the agent is supposed to call `filter_by_prompt` but the script doesn't, update that test's script to either include the filter call (if prompt is filter-shaped) or skip it (if prompt isn't). Do NOT change the test's assertions.

- [ ] **Step 8: Commit**

```bash
git add src/application/graph.py tests/application/test_graph.py
git commit -m "feat(graph): filter_by_prompt moves from ingest to finalize"
```

---

## Phase 4 — Agent prompts

### Task 4.1: Update `INGEST_SYSTEM_PROMPT` and `FINALIZE_SYSTEM_PROMPT`

**Files:**
- Modify: `src/application/agent_prompts.py`

Prompts are prose; they're reviewed in PR, not unit-tested. This is a one-commit edit.

- [ ] **Step 1: Update `src/application/agent_prompts.py`**

Find `INGEST_SYSTEM_PROMPT` and replace its value with:

```python
INGEST_SYSTEM_PROMPT = """\
You are the ingest agent. Your job is to load the input images.

1. Call `load_images` exactly once to retrieve the candidate images.
2. Stop. Do not call any tool more than necessary.
"""
```

Find `FINALIZE_SYSTEM_PROMPT` and replace its value with:

```python
FINALIZE_SYSTEM_PROMPT = """\
You are the finalize agent. Produce the final report.

Required sequence:
1. If the user prompt implies category filtering (for example "only food",
   "exclude travel", "no software"), call `filter_by_prompt` first.
   Otherwise skip to step 2.
2. Call `aggregate` on the (possibly filtered) receipts.
3. Call `detect_anomalies` on the aggregates and receipts.
4. For EACH anomaly returned by `detect_anomalies`, call `add_assumption`
   once, using the anomaly's code and message.
5. Call `generate_report`. This is REQUIRED — do not skip it. The report
   produces the user-visible final_result event.
6. Stop.

Do not call any tool after `generate_report`.
"""
```

- [ ] **Step 2: Smoke-import**

```bash
PYTHONPATH=src .venv/bin/python -c "from application.agent_prompts import INGEST_SYSTEM_PROMPT, FINALIZE_SYSTEM_PROMPT; assert 'filter_by_prompt' in FINALIZE_SYSTEM_PROMPT; assert 'filter_by_prompt' not in INGEST_SYSTEM_PROMPT; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Run full suite**

```bash
PYTHONPATH=src .venv/bin/pytest -v -m "not e2e"
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/application/agent_prompts.py
git commit -m "feat(agents): ingest prompt simplified; finalize prompt mentions filter_by_prompt"
```

---

## Phase 5 — Verification

### Task 5.1: Full suite green

- [ ] **Step 1: Run the full non-e2e suite**

```bash
PYTHONPATH=src .venv/bin/pytest -v -m "not e2e"
```

Expected: all tests pass — around 144 total (129 baseline + 8 new `filter_by_prompt` unit tests + 3 new `_parse_prompt` tests + 3 new graph tests + 1 domain test). 1 skipped (DB repositories test needing `TEST_SUPABASE_DB_URL`).

- [ ] **Step 2: Smoke-import the whole application**

```bash
PYTHONPATH=src .venv/bin/python -c "from application.graph import GraphRunner, RunState, build_graph; from application.tool_registry import filter_by_prompt, _parse_prompt, _CATEGORY_KEYWORD_MAP; from infrastructure.agent_tools import build_filter_by_prompt_tool; print('ok')"
```

Expected: `ok`.

### Task 5.2: (Optional) Live smoke — mock mode

- [ ] **Step 1: Start server in mock mode on free port**

```bash
lsof -tiTCP:8001 -sTCP:LISTEN 2>/dev/null | xargs -r kill
LLM_MODE=mock PYTHONPATH=src .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001 &
sleep 2
curl -s http://localhost:8001/health
```

Expected: `{"status":"ok","llm_mode":"mock"}`.

- [ ] **Step 2: Hit with a filter-shaped prompt**

```bash
curl -N -X POST http://localhost:8001/runs/stream \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "./assets", "prompt": "only food"}'
```

Expected: SSE stream ends with `final_result`. In mock mode the shipped `default_mock_script` doesn't include a `filter_by_prompt` call, so all receipts count (no actual filtering happens). That's OK — the path is exercised up to the point where the agent chooses whether to filter. No code path should crash.

- [ ] **Step 3: Stop the server**

```bash
lsof -tiTCP:8001 -sTCP:LISTEN 2>/dev/null | xargs -r kill
```

No commit for this task.

### Task 5.3: (Optional) Live smoke — real mode

Requires credentials in `.env` + reachable Supabase.

- [ ] **Step 1: Start server in real mode**

```bash
LLM_MODE=real PYTHONPATH=src .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001 &
sleep 2
curl -s http://localhost:8001/health
```

Expected: `{"status":"ok","llm_mode":"real"}`.

- [ ] **Step 2: Hit with a filter-shaped prompt against real assets**

```bash
curl -N -X POST http://localhost:8001/runs/stream \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "./assets", "prompt": "only food"}'
```

Expected: `tool_call(filter_by_prompt)` appears between `finalize_start` and `tool_call(aggregate)`; `final_result.total_spend` reflects only the MEALS-categorized receipts (~1 of the 18 assets fits that category). Filtered receipts appear in `receipts[]` with `status="filtered"`.

---

## Post-implementation checklist

- [ ] `make test` → all pass (filter unit tests + parse_prompt + graph filter tests + domain test).
- [ ] `make run` → `/health` responds.
- [ ] `curl /runs/stream` with `"prompt": "only food"` in real mode → `final_result.total_spend` ≤ total of all receipts; at least one receipt in `final_result.receipts` has `status="filtered"` (given a mixed-category asset folder).
- [ ] SSE wire contract unchanged (no new event types; `Receipt.status` literal widened).
- [ ] `FilterResult` class is gone from `tool_registry.py`.
- [ ] `_capture_filter` helper is gone from `graph.py`.
- [ ] `ingest_node`'s tool list is `[wrapped_load]` only.
- [ ] `finalize_node`'s tool list is `[filter_tool, aggregate_tool, detect_tool, add_assumption_tool, generate_report_tool]`.
- [ ] `INGEST_SYSTEM_PROMPT` does NOT mention filtering; `FINALIZE_SYSTEM_PROMPT` does.

---

## Plan Self-Review

**Spec coverage check:**

| Spec section | Implementing task(s) |
|---|---|
| §3.1 pipeline change | 3.1 (ingest drops filter; finalize gains it) |
| §3.2 preserved invariants | verified by 5.1 suite |
| §3.3 receipt_result vs final_result status mismatch | accepted and documented in spec; no test needed since it's emergent behavior from existing per_receipt_node |
| §4.1 `_CATEGORY_KEYWORD_MAP` | 1.2 |
| §4.2 `_parse_prompt` | 1.2 (helper + 8 parametric tests) |
| §4.3 `filter_by_prompt` tool | 2.1 (tool rewrite + 8 new tests) |
| §4.4 `Receipt.status` literal | 1.1 |
| §4.5 `build_filter_by_prompt_tool` signature | 2.1 |
| §4.6 graph wiring | 3.1 |
| §4.7 agent prompts | 4.1 |
| §5 data flow | verified by 3.1 integration test |
| §6 `filtered_by_prompt` Issue | 2.1 (tested in `test_filter_by_prompt_filtered_receipts_have_issue`) |
| §7 testing | 1.2, 2.1, 3.1 |

All spec sections have tasks.

**Placeholder scan:** none. All code blocks are complete; all commands are concrete; all expected outputs are specific.

**Type consistency:**
- `filter_by_prompt` signature — `(ctx, *, receipts: list[Receipt], user_prompt: str | None) -> list[Receipt]` — consistent across Task 2.1 and tests.
- `build_filter_by_prompt_tool` signature — kwarg-only, takes `receipts_provider: Callable[[], list[Receipt]]` — consistent in spec §4.5 and Task 2.1.
- `_capture_filtered_receipts` writes to `receipts_holder["receipts"]` — consistent with `receipts_provider=lambda: list(receipts_holder["receipts"])` used by the three downstream tools.
- `_CATEGORY_KEYWORD_MAP` keys are `AllowedCategory` enums — consistent across definition, `_parse_prompt`, and filter tests.
- `Issue(severity="warning", code="filtered_by_prompt", ...)` — consistent across spec §6 and Task 2.1.

No other issues found. Plan is ready for execution.
