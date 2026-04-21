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
async def test_ingest_node_with_prompt_filter_drops_non_matching():
    images = [_img("restaurant.png"), _img("uber.png")]
    script = [
        tool_call("load_images", {}),
        tool_call("filter_by_prompt", {}),
        finish(),
    ]
    r = _runner(prompt="only food", images=images, script=script)
    state = await r.ingest_node(RunState())
    assert [i.source_ref for i in state.images] == ["restaurant.png"]
    assert len(state.filtered_out) == 1
    assert state.filtered_out[0][0] == "uber.png"


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
