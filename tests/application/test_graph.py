import pytest
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
from domain.models import AllowedCategory, Categorization, RawReceipt
from application.graph import GraphRunner, build_graph, RunState
from application.event_bus import InMemoryEventBus
from application.traced_tool import ToolContext
from application.ports import ImageRef
from tests.fakes.mock_ocr import MockOCR
from tests.fakes.mock_llm import MockLLM
from tests.fakes.mock_image_loader import MockImageLoader
from tests.fakes.in_memory_repos import InMemoryReportRepository, InMemoryTraceRepository


class _NullTracer:
    def start_span(self, name, input=None):
        class _S:
            def end(self_, output=None, error=None): pass
        return _S()


def _refs(n=2):
    return [
        ImageRef(source_ref=f"r{i}.png", local_path=Path(f"/t/r{i}.png"))
        for i in range(1, n + 1)
    ]


def _runner_with_mocks(refs, ocr=None, llm=None, prompt=None):
    bus = InMemoryEventBus()
    events: list[dict] = []

    async def capture(e):
        events.append(e)

    bus.subscribe(capture)
    return GraphRunner(
        run_id=uuid4(),
        prompt=prompt,
        bus=bus,
        tracer=_NullTracer(),
        image_loader=MockImageLoader(refs),
        ocr=ocr or MockOCR(),
        llm=llm or MockLLM(default_category=AllowedCategory.TRAVEL),
        report_repo=InMemoryReportRepository(),
    ), events


@pytest.mark.asyncio
async def test_happy_path_event_order():
    runner, events = _runner_with_mocks(_refs(2))
    app = build_graph(runner)
    await app.ainvoke(RunState(receipts=[], current=0, errors=[], issues=[]))

    kinds = [e["event_type"] for e in events]
    assert kinds[0] == "run_started"
    assert "load_images" in str(events[1:3])
    assert kinds.count("receipt_result") == 2
    assert kinds[-1] == "final_result"


@pytest.mark.asyncio
async def test_final_result_contains_aggregates():
    llm = MockLLM(default_category=AllowedCategory.TRAVEL)
    runner, events = _runner_with_mocks(_refs(2), llm=llm)
    app = build_graph(runner)
    await app.ainvoke(RunState(receipts=[], current=0, errors=[], issues=[]))
    final = [e for e in events if e["event_type"] == "final_result"][0]
    assert final["total_spend"] == "24.68"
    assert final["by_category"]["Travel"] == "24.68"


@pytest.mark.asyncio
async def test_prompt_threads_through_to_subagent():
    llm = MockLLM(default_category=AllowedCategory.TRAVEL)
    runner, _ = _runner_with_mocks(_refs(1), llm=llm, prompt="be conservative")
    app = build_graph(runner)
    await app.ainvoke(RunState(receipts=[], current=0, errors=[], issues=[]))
    assert llm.calls[0].user_prompt == "be conservative"


# R2: run-level assumption strings
@pytest.mark.asyncio
async def test_final_result_includes_run_level_assumptions():
    runner, events = _runner_with_mocks(_refs(1))
    await build_graph(runner).ainvoke(RunState(receipts=[], current=0, errors=[], issues=[]))
    final = [e for e in events if e["event_type"] == "final_result"][0]
    codes = [i["code"] for i in final["issues_and_assumptions"]]
    assert "only_allowed_extensions" in codes
    assert "default_currency_usd" in codes
    assert "errored_receipts_excluded" in codes
