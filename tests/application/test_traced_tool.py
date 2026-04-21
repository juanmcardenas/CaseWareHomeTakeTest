import asyncio
import pytest
from uuid import uuid4
from application.traced_tool import traced_tool, ToolContext
from application.event_bus import InMemoryEventBus


class FakeTracer:
    def __init__(self):
        self.spans = []

    def start_span(self, name, input=None):
        span = FakeSpan(name, input)
        self.spans.append(span)
        return span


class FakeSpan:
    def __init__(self, name, input):
        self.name = name
        self.input = input
        self.output = None
        self.error = None
        self.ended = False

    def end(self, output=None, error=None):
        self.output = output
        self.error = error
        self.ended = True


@pytest.fixture
def ctx():
    bus = InMemoryEventBus()
    events: list[dict] = []

    async def capture(e):
        events.append(e)

    bus.subscribe(capture)
    return ToolContext(
        run_id=uuid4(),
        bus=bus,
        tracer=FakeTracer(),
        seq_counter=iter(range(1, 1000)),
    ), events


@pytest.mark.asyncio
async def test_emits_tool_call_and_tool_result_on_success(ctx):
    c, events = ctx

    @traced_tool("load_images")
    async def impl(ctx, *, folder_path):
        return {"count": 2}

    out = await impl(c, folder_path="/tmp")
    assert out == {"count": 2}
    kinds = [e["event_type"] for e in events]
    assert kinds == ["tool_call", "tool_result"]
    assert events[0]["tool"] == "load_images"
    assert events[0]["args"] == {"folder_path": "/tmp"}
    assert events[1]["error"] is False
    assert events[1]["duration_ms"] is not None
    assert events[1]["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_emits_tool_result_with_error_flag_on_exception(ctx):
    c, events = ctx

    @traced_tool("extract_receipt_fields")
    async def impl(ctx, *, image_ref):
        raise RuntimeError("OCR timed out")

    with pytest.raises(RuntimeError):
        await impl(c, image_ref="r.png")

    kinds = [e["event_type"] for e in events]
    assert kinds == ["tool_call", "tool_result"]
    assert events[1]["error"] is True
    assert "OCR timed out" in events[1]["error_message"]


@pytest.mark.asyncio
async def test_result_summary_builder_is_honored(ctx):
    c, events = ctx

    def summarize(result):
        return {"count": result["count"], "kind": "summary"}

    @traced_tool("load_images", summarize=summarize)
    async def impl(ctx, *, folder_path):
        return {"count": 5, "files": ["a", "b", "c", "d", "e"]}

    await impl(c, folder_path="/tmp")
    assert events[1]["result_summary"] == {"count": 5, "kind": "summary"}


# Revision R1: retry support
@pytest.mark.asyncio
async def test_retries_once_on_network_error(ctx):
    c, events = ctx
    attempts = {"n": 0}

    @traced_tool("extract_receipt_fields", retries=1, retry_delays_s=(0.0,))
    async def impl(ctx):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise asyncio.TimeoutError("first attempt times out")
        return {"ok": True}

    await impl(c)
    assert attempts["n"] == 2
    # Two tool_call events with attempt=1 and attempt=2
    calls = [e for e in events if e["event_type"] == "tool_call"]
    assert [e["attempt"] for e in calls] == [1, 2]


@pytest.mark.asyncio
async def test_does_not_retry_on_non_network_error(ctx):
    c, events = ctx
    attempts = {"n": 0}

    @traced_tool("categorize_receipt", retries=1)
    async def impl(ctx):
        attempts["n"] += 1
        raise ValueError("validation error, not retryable")

    with pytest.raises(ValueError):
        await impl(c)
    assert attempts["n"] == 1  # no retry
