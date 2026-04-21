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


# ---------------------------------------------------------------------------
# re_extract_with_hint tests
# ---------------------------------------------------------------------------
from domain.models import RawReceipt
from application.tool_registry import re_extract_with_hint


@pytest.mark.asyncio
async def test_re_extract_with_hint_calls_ocr_with_hint():
    class HintRecordingOCR(MockOCR):
        def __init__(self):
            super().__init__()
            self.last_hint: str | None = None
            self.call_count = 0

        async def extract(self, image, hint=None):
            self.last_hint = hint
            self.call_count += 1
            return RawReceipt(source_ref=image.source_ref, vendor="X")

    ocr = HintRecordingOCR()
    img = _img("a.png")
    r = await re_extract_with_hint(_fctx(), ocr=ocr, image=img, hint="focus on total")
    assert ocr.last_hint == "focus on total"
    assert ocr.call_count == 1
    assert r.vendor == "X"


# ---------------------------------------------------------------------------
# skip_receipt tests
# ---------------------------------------------------------------------------
from application.tool_registry import skip_receipt


@pytest.mark.asyncio
async def test_skip_receipt_returns_error_receipt_with_issue():
    rid = uuid4()
    r = await skip_receipt(_fctx(), receipt_id=rid, reason="ocr_twice_failed")
    assert r.id == rid
    assert r.status == "error"
    assert r.error == "ocr_twice_failed"
    assert len(r.issues) == 1
    assert r.issues[0].severity == "receipt_error"
    assert r.issues[0].code == "agent_skipped"
    assert r.issues[0].message == "ocr_twice_failed"
    assert r.issues[0].receipt_id == rid


# ---------------------------------------------------------------------------
# detect_anomalies tests
# ---------------------------------------------------------------------------
from domain.models import Aggregates, Anomaly
from application.tool_registry import detect_anomalies


def _r(total: Decimal, currency: str = "USD", receipt_date=None, status: str = "ok") -> Receipt:
    return Receipt(
        id=uuid4(),
        source_ref="x",
        total=total,
        currency=currency,
        receipt_date=receipt_date,
        category=AllowedCategory.OTHER,
        confidence=0.9,
        notes="x",
        status=status,
    )


@pytest.mark.asyncio
async def test_detect_anomalies_single_receipt_dominant():
    from datetime import date
    aggregates = Aggregates(total_spend=Decimal("100.00"), by_category={"Other": Decimal("100.00")})
    receipts = [_r(Decimal("85.00"), receipt_date=date(2024, 1, 1)),
                _r(Decimal("15.00"), receipt_date=date(2024, 1, 2))]
    result = await detect_anomalies(_fctx(), aggregates=aggregates, receipts=receipts)
    codes = {a.code for a in result}
    assert "single_receipt_dominant" in codes


@pytest.mark.asyncio
async def test_detect_anomalies_currency_mix():
    from datetime import date
    aggregates = Aggregates(total_spend=Decimal("20.00"), by_category={"Other": Decimal("20.00")})
    receipts = [_r(Decimal("10.00"), currency="USD", receipt_date=date(2024, 1, 1)),
                _r(Decimal("10.00"), currency="EUR", receipt_date=date(2024, 1, 2))]
    result = await detect_anomalies(_fctx(), aggregates=aggregates, receipts=receipts)
    codes = {a.code for a in result}
    assert "currency_mix" in codes


@pytest.mark.asyncio
async def test_detect_anomalies_many_missing_dates():
    aggregates = Aggregates(total_spend=Decimal("20.00"), by_category={"Other": Decimal("20.00")})
    receipts = [_r(Decimal("10.00"), receipt_date=None),
                _r(Decimal("10.00"), receipt_date=None)]
    result = await detect_anomalies(_fctx(), aggregates=aggregates, receipts=receipts)
    codes = {a.code for a in result}
    assert "many_missing_dates" in codes


@pytest.mark.asyncio
async def test_detect_anomalies_clean_run_returns_empty():
    from datetime import date
    aggregates = Aggregates(total_spend=Decimal("100.00"), by_category={"Other": Decimal("100.00")})
    receipts = [_r(Decimal("40.00"), receipt_date=date(2024, 1, 1)),
                _r(Decimal("35.00"), receipt_date=date(2024, 1, 2)),
                _r(Decimal("25.00"), receipt_date=date(2024, 1, 3))]
    result = await detect_anomalies(_fctx(), aggregates=aggregates, receipts=receipts)
    assert result == []


# ---------------------------------------------------------------------------
# add_assumption
# ---------------------------------------------------------------------------
from application.tool_registry import add_assumption


@pytest.mark.asyncio
async def test_add_assumption_returns_warning_issue():
    iss = await add_assumption(_fctx(), code="review_currency_mix", message="Multiple currencies detected")
    assert iss.severity == "warning"
    assert iss.code == "review_currency_mix"
    assert iss.message == "Multiple currencies detected"
    assert iss.receipt_id is None
