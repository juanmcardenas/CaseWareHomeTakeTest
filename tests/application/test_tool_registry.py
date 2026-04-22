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
from uuid import uuid4 as _uuid4_filter
from decimal import Decimal as _Decimal_filter
from domain.models import AllowedCategory as _AC
from domain.models import Receipt as _Receipt_filter, Issue as _Issue_filter
from application.tool_registry import filter_by_prompt as _filter_by_prompt


def _ok_receipt(category: _AC, total: str = "10.00", source_ref: str = "x") -> _Receipt_filter:
    return _Receipt_filter(
        id=_uuid4_filter(), source_ref=source_ref, status="ok",
        category=category, confidence=0.9, notes="n",
        total=_Decimal_filter(total), currency="USD",
    )


def _error_receipt(source_ref: str = "y") -> _Receipt_filter:
    return _Receipt_filter(id=_uuid4_filter(), source_ref=source_ref, status="error", error="boom")


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
    assert out_by_ref["e"].status == "error"
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


# ---------------------------------------------------------------------------
# builder smoke test
# ---------------------------------------------------------------------------
from langchain_core.tools import BaseTool
from infrastructure.agent_tools import build_load_images_tool


@pytest.mark.asyncio
async def test_build_load_images_tool_returns_base_tool_usable_by_agent():
    loader = MockImageLoader([_img("a.png"), _img("b.png")])
    ctx_factory = lambda: _fctx()
    tool = build_load_images_tool(ctx_factory=ctx_factory, loader=loader)
    assert isinstance(tool, BaseTool)
    assert tool.name == "load_images"
    result = await tool.ainvoke({})
    assert isinstance(result, list)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# _parse_prompt tests
# ---------------------------------------------------------------------------
from application.tool_registry import _parse_prompt


@pytest.mark.parametrize("prompt,expected_exclude", [
    ("exclude travel", {_AC.TRAVEL}),
    ("except food", {_AC.MEALS}),
    ("not office", {_AC.OFFICE_SUPPLIES}),
    ("no travel please", {_AC.TRAVEL}),
    ("without software", {_AC.SOFTWARE}),
    ("skip utilities", {_AC.UTILITIES}),
])
def test_parse_prompt_detects_exclusion_on_each_negation_word(prompt, expected_exclude):
    include, exclude = _parse_prompt(prompt)
    assert include == set()
    assert exclude == expected_exclude


def test_parse_prompt_include_is_default_when_no_negation():
    include, exclude = _parse_prompt("only food")
    assert exclude == set()
    assert include == {_AC.MEALS}


def test_parse_prompt_multiple_categories():
    include, exclude = _parse_prompt("food and office supplies please")
    assert exclude == set()
    assert include == {_AC.MEALS, _AC.OFFICE_SUPPLIES}
