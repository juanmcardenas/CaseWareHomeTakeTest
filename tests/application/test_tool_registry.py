import pytest
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
from itertools import count
from domain.models import (
    AllowedCategory, Categorization, Issue, NormalizedReceipt, RawReceipt, Receipt,
)
from application.ports import ImageRef
from application.event_bus import InMemoryEventBus
from application.traced_tool import ToolContext
from application.tool_registry import (
    load_images, extract_receipt_fields, normalize_receipt,
    categorize_receipt, aggregate_receipts, generate_report,
)
from tests.fakes.mock_ocr import MockOCR
from tests.fakes.mock_llm import MockLLM
from tests.fakes.mock_image_loader import MockImageLoader
from infrastructure.tracing.json_logs_adapter import JSONLogsTracer


class _NullTracer:
    def start_span(self, name, input=None):
        class _S:
            def end(self_, output=None, error=None): pass
        return _S()


def _ctx(bus):
    return ToolContext(run_id=uuid4(), bus=bus, tracer=_NullTracer(),
                       seq_counter=iter(range(1, 1000)))


def _fctx():
    return ToolContext(
        run_id=uuid4(),
        bus=InMemoryEventBus(),
        tracer=_NullTracer(),
        seq_counter=count(1),
        receipt_id=None,
    )


def _img(name: str) -> ImageRef:
    return ImageRef(source_ref=name, local_path=Path(f"/tmp/{name}"))


@pytest.mark.asyncio
async def test_load_images_returns_refs_and_emits_tool_pair():
    bus = InMemoryEventBus()
    events = []

    async def capture(e):
        events.append(e)

    bus.subscribe(capture)
    loader = MockImageLoader([ImageRef(source_ref="a.png", local_path=Path("/t/a.png"))])
    refs = await load_images(_ctx(bus), loader=loader)
    assert len(refs) == 1
    assert [e["event_type"] for e in events] == ["tool_call", "tool_result"]


@pytest.mark.asyncio
async def test_extract_receipt_fields_calls_ocr():
    bus = InMemoryEventBus()
    ocr = MockOCR(responses={"a.png": RawReceipt(source_ref="a.png", vendor="V", total_raw="$1.00")})
    raw = await extract_receipt_fields(_ctx(bus), ocr=ocr,
                                       image=ImageRef(source_ref="a.png", local_path=Path("/t/a.png")))
    assert raw.vendor == "V"


@pytest.mark.asyncio
async def test_normalize_receipt_returns_normalized():
    bus = InMemoryEventBus()
    raw = RawReceipt(source_ref="a.png", vendor="V", total_raw="$45.67",
                     receipt_date="2024-03-15")
    n = await normalize_receipt(_ctx(bus), raw=raw)
    assert n.total == Decimal("45.67")
    assert n.currency == "USD"


@pytest.mark.asyncio
async def test_categorize_receipt_calls_llm_with_prompt():
    bus = InMemoryEventBus()
    llm = MockLLM(default_category=AllowedCategory.TRAVEL)
    n = NormalizedReceipt(source_ref="a.png", vendor="V", total=Decimal("10"))
    cat = await categorize_receipt(_ctx(bus), llm=llm, normalized=n, user_prompt="test")
    assert cat.category == AllowedCategory.TRAVEL
    assert llm.calls[0].user_prompt == "test"


@pytest.mark.asyncio
async def test_aggregate_receipts():
    bus = InMemoryEventBus()
    receipts = [
        Receipt(id=uuid4(), source_ref="a.png", category=AllowedCategory.TRAVEL,
                total=Decimal("45.67"), status="ok"),
    ]
    agg = await aggregate_receipts(_ctx(bus), receipts=receipts)
    assert agg.total_spend == Decimal("45.67")


@pytest.mark.asyncio
async def test_generate_report_bundles_fields():
    bus = InMemoryEventBus()
    rid = uuid4()
    receipts = []
    from domain.aggregation import aggregate
    agg = aggregate([])
    rep = await generate_report(_ctx(bus),
                                run_id=rid, aggregates=agg,
                                receipts=receipts, issues=[])
    assert rep.run_id == rid
    assert rep.total_spend == Decimal("0.00")


# ---------------------------------------------------------------------------
# filter_by_prompt tests
# ---------------------------------------------------------------------------
from application.tool_registry import filter_by_prompt, FilterResult


@pytest.mark.asyncio
async def test_filter_by_prompt_no_prompt_keeps_all():
    imgs = [_img("a.png"), _img("b.png")]
    r = await filter_by_prompt(_fctx(), images=imgs, user_prompt=None)
    assert r.kept == imgs
    assert r.dropped == []


@pytest.mark.asyncio
async def test_filter_by_prompt_unknown_keyword_keeps_all():
    imgs = [_img("a.png"), _img("b.png")]
    r = await filter_by_prompt(_fctx(), images=imgs, user_prompt="arbitrary freeform text")
    assert r.kept == imgs
    assert r.dropped == []


@pytest.mark.asyncio
async def test_filter_by_prompt_food_keyword_matches_restaurant_filename():
    imgs = [_img("restaurant_001.png"), _img("uber_receipt.png"), _img("cafe_drink.png")]
    r = await filter_by_prompt(_fctx(), images=imgs, user_prompt="only food")
    kept_names = {i.source_ref for i in r.kept}
    assert kept_names == {"restaurant_001.png", "cafe_drink.png"}
    assert len(r.dropped) == 1
    assert r.dropped[0][0] == "uber_receipt.png"
    assert "food" in r.dropped[0][1].lower() or "keyword" in r.dropped[0][1].lower()
