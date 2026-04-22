"""
Agentic graph tests — new tests live here during TDD; this file will be
renamed to test_graph.py in Phase 7.8 after the old tests are deleted.
"""
from __future__ import annotations
import pytest
from pathlib import Path
from uuid import uuid4
from decimal import Decimal

from application.graph import GraphRunner, RunState
from application.ports import ImageRef
from domain.models import RawReceipt, NormalizedReceipt, Categorization, AllowedCategory, Receipt
from tests.fakes.fake_chat_model import FakeChatModelAdapter, tool_call, finish
from tests.fakes.mock_image_loader import MockImageLoader
from tests.fakes.mock_ocr import MockOCR
from tests.fakes.mock_llm import MockLLM
from tests.fakes.in_memory_repos import InMemoryEventBus, InMemoryReportRepository


class _NullTracer:
    def start_span(self, name, input=None):
        class _S:
            def end(self, output=None, error=None): pass
        return _S()


def _img(name: str) -> ImageRef:
    return ImageRef(source_ref=name, local_path=Path(f"/tmp/{name}"))


def _runner(*, prompt=None, images, script):
    return GraphRunner(
        run_id=uuid4(),
        prompt=prompt,
        bus=InMemoryEventBus(),
        tracer=_NullTracer(),
        image_loader=MockImageLoader(images),
        ocr=MockOCR(),
        llm=MockLLM(),
        chat_model_port=FakeChatModelAdapter(script),
        report_repo=InMemoryReportRepository(),
    )


@pytest.mark.asyncio
async def test_ingest_node_happy_path_populates_state_images():
    images = [_img("a.png"), _img("b.png")]
    script = [
        tool_call("load_images", {}),
        finish(),
    ]
    r = _runner(images=images, script=script)
    state = await r.ingest_node(RunState())
    assert len(state.images) == 2
    assert state.filtered_out == []


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
    assert state.filtered_out == []


@pytest.mark.asyncio
async def test_ingest_node_kept_images_have_path_not_string_local_path():
    """load_images returns ImageRefs whose local_path must remain a Path object
    (not a string) so that downstream OCR calls image.local_path.read_bytes()
    without error."""
    images = [_img("restaurant.png"), _img("uber.png")]
    script = [
        tool_call("load_images", {}),
        finish(),
    ]
    r = _runner(prompt="only food", images=images, script=script)
    state = await r.ingest_node(RunState())
    assert len(state.images) == 2
    for img in state.images:
        assert isinstance(img.local_path, Path), (
            f"local_path must be Path, got {type(img.local_path).__name__}: "
            f"{img.local_path!r}"
        )


@pytest.mark.asyncio
async def test_ingest_node_empty_returns_state_with_no_images():
    script = [
        tool_call("load_images", {}),
        finish(),
    ]
    r = _runner(images=[], script=script)
    state = await r.ingest_node(RunState())
    assert state.images == []
    assert state.filtered_out == []


@pytest.mark.asyncio
async def test_per_receipt_node_happy_path_produces_ok_receipt():
    images = [_img("a.png")]
    ocr = MockOCR(responses={"a.png": RawReceipt(
        source_ref="a.png", vendor="Acme", receipt_date="2024-03-01",
        total_raw="$50.00", ocr_confidence=0.95,
    )})
    llm = MockLLM(default_category=AllowedCategory.MEALS)
    script = [
        tool_call("extract_receipt_fields", {}),
        tool_call("normalize_receipt", {}),
        tool_call("categorize_receipt", {}),
        finish(),
    ]
    r = GraphRunner(
        run_id=uuid4(), prompt=None,
        bus=InMemoryEventBus(), tracer=_NullTracer(),
        image_loader=MockImageLoader(images), ocr=ocr, llm=llm,
        chat_model_port=FakeChatModelAdapter(script),
        report_repo=InMemoryReportRepository(),
    )
    state = RunState(images=images, current=0)
    state = await r.per_receipt_node(state)
    assert state.current == 1
    assert len(state.receipts) == 1
    receipt = state.receipts[0]
    assert receipt.status == "ok"
    assert receipt.vendor == "Acme"
    assert receipt.category == AllowedCategory.MEALS


@pytest.mark.asyncio
async def test_per_receipt_node_agent_skip_produces_error_receipt():
    images = [_img("a.png")]
    script = [
        tool_call("skip_receipt", {"reason": "bad_image"}),
        finish(),
    ]
    r = _runner(images=images, script=script)
    state = RunState(images=images, current=0)
    state = await r.per_receipt_node(state)
    assert state.receipts[0].status == "error"
    assert state.receipts[0].error == "bad_image"


@pytest.mark.asyncio
async def test_per_receipt_node_agent_finishes_early_produces_error_receipt():
    images = [_img("a.png")]
    script = [finish()]  # agent gives up without doing anything
    r = _runner(images=images, script=script)
    state = RunState(images=images, current=0)
    state = await r.per_receipt_node(state)
    assert state.receipts[0].status == "error"
    assert state.receipts[0].error == "agent_did_not_finish"


@pytest.mark.asyncio
async def test_per_receipt_node_tool_error_is_labeled_tool_failed():
    """When a tool raises inside the agent's loop and the agent gives up,
    the synthesized error receipt should carry the actual tool error —
    not the generic 'agent_did_not_finish' label."""
    images = [_img("a.png")]
    # OCR succeeds, normalize will fail on genuinely unparseable date garbage
    ocr = MockOCR(responses={"a.png": RawReceipt(
        source_ref="a.png", vendor="Acme",
        receipt_date="20.5 10 02 18 24:02 05 00",  # OCR noise, unparseable
        total_raw="$10", ocr_confidence=0.9,
    )})
    script = [
        tool_call("extract_receipt_fields", {}),
        tool_call("normalize_receipt", {}),  # will raise ValueError
        finish(),  # agent gives up after seeing the error
    ]
    r = GraphRunner(
        run_id=uuid4(), prompt=None,
        bus=InMemoryEventBus(), tracer=_NullTracer(),
        image_loader=MockImageLoader(images), ocr=ocr, llm=MockLLM(),
        chat_model_port=FakeChatModelAdapter(script),
        report_repo=InMemoryReportRepository(),
    )
    state = RunState(images=images, current=0)
    state = await r.per_receipt_node(state)
    receipt = state.receipts[0]
    assert receipt.status == "error"
    # The actual tool error should surface, not agent_did_not_finish
    assert "unparseable date" in (receipt.error or ""), f"got: {receipt.error}"
    assert receipt.issues[0].code == "tool_failed", f"got: {receipt.issues[0].code}"


@pytest.mark.asyncio
async def test_finalize_node_all_receipts_errored_emits_run_level_error():
    bus = InMemoryEventBus()
    r = GraphRunner(
        run_id=uuid4(), prompt=None,
        bus=bus, tracer=_NullTracer(),
        image_loader=MockImageLoader([]), ocr=MockOCR(), llm=MockLLM(),
        chat_model_port=FakeChatModelAdapter([]),  # agent NOT invoked
        report_repo=InMemoryReportRepository(),
    )
    errored = Receipt(id=uuid4(), source_ref="a.png", status="error", error="x")
    state = RunState(receipts=[errored])
    state = await r.finalize_node(state)
    codes = [e.get("code") for e in bus.published if e.get("event_type") == "error"]
    assert "all_receipts_failed" in codes


@pytest.mark.asyncio
async def test_finalize_node_happy_path_emits_final_result():
    images = [_img("a.png")]
    script = [
        tool_call("aggregate", {}),
        tool_call("detect_anomalies", {}),
        tool_call("generate_report", {}),
        finish(),
    ]
    ok = Receipt(
        id=uuid4(), source_ref="a.png", status="ok",
        category=AllowedCategory.MEALS, confidence=0.9, notes="x",
        total=Decimal("10.00"), currency="USD",
    )
    bus = InMemoryEventBus()
    r = GraphRunner(
        run_id=uuid4(), prompt=None,
        bus=bus, tracer=_NullTracer(),
        image_loader=MockImageLoader(images), ocr=MockOCR(), llm=MockLLM(),
        chat_model_port=FakeChatModelAdapter(script),
        report_repo=InMemoryReportRepository(),
    )
    state = RunState(receipts=[ok])
    state = await r.finalize_node(state)
    event_types = [e.get("event_type") for e in bus.published]
    assert "final_result" in event_types


@pytest.mark.asyncio
async def test_finalize_node_missing_generate_report_emits_no_final_report_error():
    ok = Receipt(id=uuid4(), source_ref="a.png", status="ok", total=Decimal("10"), currency="USD",
                 category=AllowedCategory.OTHER, confidence=0.8, notes="x")
    bus = InMemoryEventBus()
    script = [tool_call("aggregate", {}), finish()]  # skips generate_report
    r = GraphRunner(
        run_id=uuid4(), prompt=None,
        bus=bus, tracer=_NullTracer(),
        image_loader=MockImageLoader([]), ocr=MockOCR(), llm=MockLLM(),
        chat_model_port=FakeChatModelAdapter(script),
        report_repo=InMemoryReportRepository(),
    )
    state = RunState(receipts=[ok])
    state = await r.finalize_node(state)
    codes = [e.get("code") for e in bus.published if e.get("event_type") == "error"]
    assert "no_final_report" in codes


@pytest.mark.asyncio
async def test_full_graph_happy_path_two_receipts():
    images = [_img("a.png"), _img("b.png")]
    ocr = MockOCR(responses={
        "a.png": RawReceipt(source_ref="a.png", vendor="Acme", receipt_date="2024-03-01",
                            total_raw="$50.00", ocr_confidence=0.95),
        "b.png": RawReceipt(source_ref="b.png", vendor="Bravo", receipt_date="2024-03-02",
                            total_raw="$30.00", ocr_confidence=0.95),
    })
    llm = MockLLM(default_category=AllowedCategory.MEALS)

    # Script: ingest (load_images+finish) + 2×per_receipt (extract+normalize+categorize+finish) + finalize (aggregate+detect+generate_report+finish)
    script = [
        tool_call("load_images", {}), finish(),
        tool_call("extract_receipt_fields", {}),
        tool_call("normalize_receipt", {}),
        tool_call("categorize_receipt", {}),
        finish(),
        tool_call("extract_receipt_fields", {}),
        tool_call("normalize_receipt", {}),
        tool_call("categorize_receipt", {}),
        finish(),
        tool_call("aggregate", {}),
        tool_call("detect_anomalies", {}),
        tool_call("generate_report", {}),
        finish(),
    ]
    bus = InMemoryEventBus()
    r = GraphRunner(
        run_id=uuid4(), prompt=None,
        bus=bus, tracer=_NullTracer(),
        image_loader=MockImageLoader(images), ocr=ocr, llm=llm,
        chat_model_port=FakeChatModelAdapter(script),
        report_repo=InMemoryReportRepository(),
    )
    from application.graph import build_graph
    graph = build_graph(r)
    await graph.ainvoke(RunState())
    event_types = [e.get("event_type") for e in bus.published]
    assert event_types[0] == "run_started"
    assert event_types[-1] == "final_result"
    assert event_types.count("receipt_result") == 2


@pytest.mark.asyncio
async def test_full_graph_zero_images_emits_no_images():
    script = [tool_call("load_images", {}), finish()]
    bus = InMemoryEventBus()
    r = GraphRunner(
        run_id=uuid4(), prompt=None,
        bus=bus, tracer=_NullTracer(),
        image_loader=MockImageLoader([]), ocr=MockOCR(), llm=MockLLM(),
        chat_model_port=FakeChatModelAdapter(script),
        report_repo=InMemoryReportRepository(),
    )
    from application.graph import build_graph
    graph = build_graph(r)
    await graph.ainvoke(RunState())
    codes = [e.get("code") for e in bus.published if e.get("event_type") == "error"]
    assert "no_images" in codes


@pytest.mark.asyncio
async def test_full_graph_no_images_loaded_emits_no_images_error():
    """When the agent loads no images, the graph emits no_images and terminates.
    (filter_by_prompt is no longer available in ingest; filtering happens in finalize.)"""
    script = [tool_call("load_images", {}), finish()]
    bus = InMemoryEventBus()
    r = GraphRunner(
        run_id=uuid4(), prompt="only food",
        bus=bus, tracer=_NullTracer(),
        image_loader=MockImageLoader([]), ocr=MockOCR(), llm=MockLLM(),
        chat_model_port=FakeChatModelAdapter(script),
        report_repo=InMemoryReportRepository(),
    )
    from application.graph import build_graph
    graph = build_graph(r)
    await graph.ainvoke(RunState())
    codes = [e.get("code") for e in bus.published if e.get("event_type") == "error"]
    assert "no_images" in codes


@pytest.mark.asyncio
async def test_ingest_node_infinite_loop_triggers_iterations_exhausted():
    """
    Agent that loops calling load_images forever hits LangGraph's internal routing
    error (KeyError: 'model') because the inner create_agent subgraph cannot route
    back to the model node once its conditional-edge map is exhausted by the cycling
    FakeMessagesListChatModel. This KeyError is a subclass of Exception, so the
    wrapper's except-block catches it and emits ingest_iterations_exhausted.
    This is the implicit iteration cap — the safety net works through the except
    Exception clause rather than a per-node max_iterations kwarg.
    """
    images = [_img("a.png")]
    # Script with only tool_calls and no finish: FakeMessagesListChatModel cycles
    # back to index 0 after exhaustion, so the agent loops calling load_images
    # until LangGraph's inner agent raises an exception.
    script = [tool_call("load_images", {})] * 3
    bus = InMemoryEventBus()
    r = GraphRunner(
        run_id=uuid4(), prompt=None,
        bus=bus, tracer=_NullTracer(),
        image_loader=MockImageLoader(images), ocr=MockOCR(), llm=MockLLM(),
        chat_model_port=FakeChatModelAdapter(script),
        report_repo=InMemoryReportRepository(),
    )
    state = await r.ingest_node(RunState())

    codes = [e.get("code") for e in bus.published if e.get("event_type") == "error"]
    assert "ingest_iterations_exhausted" in codes
    assert "ingest_iterations_exhausted" in state.errors


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
    assert final["total_spend"] == "25.00"
    receipts_by_ref = {rc["source_ref"]: rc for rc in final["receipts"]}
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
    assert final["total_spend"] == "125.00"
    receipts_by_ref = {rc["source_ref"]: rc for rc in final["receipts"]}
    assert receipts_by_ref["m.png"]["status"] == "ok"
    assert receipts_by_ref["t.png"]["status"] == "ok"


@pytest.mark.asyncio
async def test_full_graph_with_filter_prompt_yields_partial_aggregates():
    """End-to-end with filter: event ordering + partial totals."""
    images = [_img("a.png"), _img("b.png")]
    ocr = MockOCR(responses={
        "a.png": RawReceipt(source_ref="a.png", vendor="Cafe", receipt_date="2024-03-01",
                            total_raw="$25.00", ocr_confidence=0.95),
        "b.png": RawReceipt(source_ref="b.png", vendor="Uber", receipt_date="2024-03-02",
                            total_raw="$100.00", ocr_confidence=0.95),
    })

    class _TwoCategoryLLM(MockLLM):
        async def categorize(self, normalized, allowed, user_prompt):
            from domain.models import Categorization, AllowedCategory as _AC2
            if normalized.vendor == "Cafe":
                return Categorization(category=_AC2.MEALS, confidence=0.9, notes="cafe")
            return Categorization(category=_AC2.TRAVEL, confidence=0.9, notes="uber")

    llm = _TwoCategoryLLM()
    script = [
        tool_call("load_images", {}), finish(),
        tool_call("extract_receipt_fields", {}),
        tool_call("normalize_receipt", {}),
        tool_call("categorize_receipt", {}),
        finish(),
        tool_call("extract_receipt_fields", {}),
        tool_call("normalize_receipt", {}),
        tool_call("categorize_receipt", {}),
        finish(),
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

    events = bus.published
    finalize_start_seq = next(
        e["seq"] for e in events
        if e.get("event_type") == "progress" and e.get("step") == "finalize_start"
    )
    filter_seq = next(
        e["seq"] for e in events
        if e.get("event_type") == "tool_call" and e.get("tool") == "filter_by_prompt"
    )
    aggregate_seq = next(
        e["seq"] for e in events
        if e.get("event_type") == "tool_call" and e.get("tool") == "aggregate"
    )
    assert finalize_start_seq < filter_seq < aggregate_seq

    final = [e for e in events if e.get("event_type") == "final_result"][0]
    assert final["total_spend"] == "25.00"
