"""
The 6 tools. Each is a thin wrapper decorated with @traced_tool.

Registry constraint: the application (graph) must only touch receipt data
by calling these. Direct use of adapters or domain functions outside these
wrappers bypasses the trace — don't do that.

Network-sensitive tools (extract_receipt_fields, categorize_receipt) retry
once on network-class exceptions per plan revision R1.
"""
from decimal import Decimal
from typing import Awaitable, Callable
from uuid import UUID
from pydantic import BaseModel
from domain.aggregation import aggregate as _aggregate_pure
from domain.models import (
    Aggregates, AllowedCategory, Anomaly, Categorization, Issue, NormalizedReceipt,
    RawReceipt, Receipt, Report,
)
from domain.normalization import normalize as _normalize_pure
from application.ports import ImageLoaderPort, ImageRef, LLMPort, OCRPort
from application.subagent import categorize_with_subagent
from application.traced_tool import ToolContext, traced_tool
from langchain_core.tools import StructuredTool


# 1. load_images
@traced_tool(
    "load_images",
    summarize=lambda refs: {"count": len(refs)},
)
async def load_images(ctx: ToolContext, *, loader: ImageLoaderPort) -> list[ImageRef]:
    return await loader.load()


# 2. extract_receipt_fields (network-sensitive — retries once)
def _summarize_raw(r: RawReceipt) -> dict:
    return {
        "vendor": r.vendor,
        "has_total": r.total_raw is not None,
        "ocr_confidence": r.ocr_confidence,
    }


@traced_tool("extract_receipt_fields", summarize=_summarize_raw, retries=1)
async def extract_receipt_fields(
    ctx: ToolContext, *, ocr: OCRPort, image: ImageRef,
) -> RawReceipt:
    return await ocr.extract(image)


# 3. normalize_receipt
def _summarize_normalized(n: NormalizedReceipt) -> dict:
    return {
        "vendor": n.vendor,
        "receipt_date": n.receipt_date.isoformat() if n.receipt_date else None,
        "total": str(n.total) if n.total is not None else None,
        "currency": n.currency,
    }


@traced_tool("normalize_receipt", summarize=_summarize_normalized)
async def normalize_receipt(
    ctx: ToolContext, *, raw: RawReceipt,
) -> NormalizedReceipt:
    return _normalize_pure(raw)


# 4. categorize_receipt (network-sensitive — retries once)
def _summarize_categorization(c: Categorization) -> dict:
    return {
        "category": c.category.value,
        "confidence": c.confidence,
        "issue_count": len(c.issues),
    }


@traced_tool("categorize_receipt", summarize=_summarize_categorization, retries=1)
async def categorize_receipt(
    ctx: ToolContext, *, llm: LLMPort,
    normalized: NormalizedReceipt, user_prompt: str | None,
) -> Categorization:
    return await categorize_with_subagent(llm, normalized, user_prompt)


# 5. aggregate
def _summarize_aggregates(a: Aggregates) -> dict:
    return {
        "total_spend": str(a.total_spend),
        "by_category": {k: str(v) for k, v in a.by_category.items()},
    }


@traced_tool("aggregate", summarize=_summarize_aggregates)
async def aggregate_receipts(
    ctx: ToolContext, *, receipts: list[Receipt],
) -> Aggregates:
    return _aggregate_pure(receipts)


# 6. generate_report
def _summarize_report(r: Report) -> dict:
    return {
        "total_spend": str(r.total_spend),
        "receipt_count": len(r.receipts),
        "issue_count": len(r.issues_and_assumptions),
    }


@traced_tool("generate_report", summarize=_summarize_report)
async def generate_report(
    ctx: ToolContext, *,
    run_id: UUID, aggregates: Aggregates,
    receipts: list[Receipt], issues: list[Issue],
) -> Report:
    return Report(
        run_id=run_id,
        total_spend=aggregates.total_spend,
        by_category=aggregates.by_category,
        receipts=receipts,
        issues_and_assumptions=issues,
    )


# 7. filter_by_prompt — pure-Python keyword heuristic
_PROMPT_KEYWORD_MAP: dict[str, list[str]] = {
    "food": ["restaurant", "cafe", "lunch", "dinner", "meal", "food", "coffee"],
    "travel": ["uber", "lyft", "taxi", "flight", "hotel", "airbnb", "train"],
    "office": ["office", "supplies", "staples", "paper"],
    "software": ["subscription", "saas", "stripe", "github", "aws"],
}


class FilterResult(BaseModel):
    kept: list[ImageRef]
    dropped: list[tuple[str, str]]

    model_config = {"arbitrary_types_allowed": True}


def _matched_keywords(prompt: str) -> list[str]:
    prompt_lower = prompt.lower()
    matched: list[str] = []
    for trigger, keywords in _PROMPT_KEYWORD_MAP.items():
        if trigger in prompt_lower:
            matched.extend(keywords)
    return matched


@traced_tool(
    "filter_by_prompt",
    summarize=lambda r: {"kept": len(r.kept), "dropped": len(r.dropped)},
)
async def filter_by_prompt(
    ctx: ToolContext, *, images: list[ImageRef], user_prompt: str | None,
) -> FilterResult:
    if not user_prompt:
        return FilterResult(kept=list(images), dropped=[])
    keywords = _matched_keywords(user_prompt)
    if not keywords:
        return FilterResult(kept=list(images), dropped=[])
    kept: list[ImageRef] = []
    dropped: list[tuple[str, str]] = []
    for img in images:
        name = img.source_ref.lower()
        if any(kw in name for kw in keywords):
            kept.append(img)
        else:
            dropped.append((img.source_ref, f"no keyword from prompt ({', '.join(keywords)}) in filename"))
    return FilterResult(kept=kept, dropped=dropped)


# 8. re_extract_with_hint — retries OCR with a caller-supplied hint
@traced_tool("re_extract_with_hint", summarize=_summarize_raw, retries=1)
async def re_extract_with_hint(
    ctx: ToolContext, *, ocr: OCRPort, image: ImageRef, hint: str,
) -> RawReceipt:
    return await ocr.extract(image, hint=hint)


# 9. skip_receipt — agent-driven skip
def _summarize_skip(r: Receipt) -> dict:
    return {"id": str(r.id), "reason": r.error}


@traced_tool("skip_receipt", summarize=_summarize_skip)
async def skip_receipt(
    ctx: ToolContext, *, receipt_id: UUID, reason: str,
) -> Receipt:
    return Receipt(
        id=receipt_id,
        source_ref="",
        status="error",
        error=reason,
        issues=[Issue(
            severity="receipt_error",
            code="agent_skipped",
            message=reason,
            receipt_id=receipt_id,
        )],
    )


# 10. detect_anomalies — pure rules over aggregates + receipts
def _summarize_anomalies(result: list[Anomaly]) -> dict:
    return {"count": len(result), "codes": [a.code for a in result]}


@traced_tool("detect_anomalies", summarize=_summarize_anomalies)
async def detect_anomalies(
    ctx: ToolContext, *, aggregates: Aggregates, receipts: list[Receipt],
) -> list[Anomaly]:
    out: list[Anomaly] = []
    ok_receipts = [r for r in receipts if r.status == "ok" and r.total is not None]
    total = aggregates.total_spend

    # Rule 1: single receipt >= 80% of spend
    if ok_receipts and total > 0:
        for r in ok_receipts:
            if (r.total / total) >= Decimal("0.80"):
                out.append(Anomaly(
                    code="single_receipt_dominant",
                    message=f"Receipt {r.source_ref or r.id} is {(r.total / total * 100):.0f}% of total spend",
                ))
                break

    # Rule 2: currency mix
    currencies = {r.currency for r in ok_receipts if r.currency}
    if len(currencies) > 1:
        out.append(Anomaly(
            code="currency_mix",
            message=f"Receipts contain multiple currencies: {sorted(currencies)}",
        ))

    # Rule 3: >= 50% of ok receipts missing dates
    if ok_receipts:
        missing = sum(1 for r in ok_receipts if r.receipt_date is None)
        if missing / len(ok_receipts) >= 0.5:
            out.append(Anomaly(
                code="many_missing_dates",
                message=f"{missing} of {len(ok_receipts)} receipts are missing a date",
            ))

    return out


# 11. add_assumption — agent narrates a warning into the final report
@traced_tool("add_assumption", summarize=lambda i: {"code": i.code})
async def add_assumption(
    ctx: ToolContext, *, code: str, message: str,
) -> Issue:
    return Issue(severity="warning", code=code, message=message)


TOOL_NAMES = [
    "load_images",
    "extract_receipt_fields",
    "normalize_receipt",
    "categorize_receipt",
    "aggregate",
    "generate_report",
    "filter_by_prompt",
    "re_extract_with_hint",
    "skip_receipt",
    "detect_anomalies",
    "add_assumption",
]


# ------------------- Agent-facing tool builders -------------------
# Each builder returns a LangChain StructuredTool that bakes in fixed
# dependencies and the ToolContext factory, exposing only agent-provided
# arguments to the LLM. Return values are JSON-serialized via model_dump
# for the agent's tool-observation content.


def _dump(v):
    """Convert tool results into JSON-serializable structures for the agent."""
    if v is None:
        return None
    if hasattr(v, "model_dump"):
        return v.model_dump(mode="json")
    if isinstance(v, list):
        return [_dump(x) for x in v]
    if isinstance(v, tuple):
        return [_dump(x) for x in v]
    if isinstance(v, dict):
        return {k: _dump(x) for k, x in v.items()}
    return v


def build_load_images_tool(*, ctx_factory: Callable[[], ToolContext], loader: ImageLoaderPort) -> StructuredTool:
    async def _run() -> list[dict]:
        result = await load_images(ctx_factory(), loader=loader)
        return _dump(result)

    return StructuredTool.from_function(
        coroutine=_run,
        name="load_images",
        description="Load all receipt images available for this run. Takes no arguments.",
    )


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


def build_extract_receipt_fields_tool(
    *, ctx_factory: Callable[[], ToolContext], ocr: OCRPort, image_provider: Callable[[], ImageRef],
) -> StructuredTool:
    async def _run() -> dict:
        result = await extract_receipt_fields(ctx_factory(), ocr=ocr, image=image_provider())
        return _dump(result)

    return StructuredTool.from_function(
        coroutine=_run,
        name="extract_receipt_fields",
        description="Run OCR on the current receipt image. Takes no arguments; uses the run's current image.",
    )


def build_re_extract_with_hint_tool(
    *, ctx_factory: Callable[[], ToolContext], ocr: OCRPort, image_provider: Callable[[], ImageRef],
) -> StructuredTool:
    from pydantic import BaseModel as _BM, Field as _F

    class _Args(_BM):
        hint: str = _F(..., description="A short hint appended to the OCR system prompt")

    async def _run(hint: str) -> dict:
        result = await re_extract_with_hint(ctx_factory(), ocr=ocr, image=image_provider(), hint=hint)
        return _dump(result)

    return StructuredTool.from_function(
        coroutine=_run,
        name="re_extract_with_hint",
        description="Re-run OCR on the current image with an extra hint string.",
        args_schema=_Args,
    )


def build_normalize_receipt_tool(
    *, ctx_factory: Callable[[], ToolContext], raw_holder: dict,
) -> StructuredTool:
    # raw_holder is a dict with key 'raw' set by the extract tool's capture wrapper
    async def _run() -> dict:
        raw = raw_holder.get("raw")
        if raw is None:
            raise RuntimeError("normalize_receipt called before extract_receipt_fields")
        result = await normalize_receipt(ctx_factory(), raw=raw)
        return _dump(result)

    return StructuredTool.from_function(
        coroutine=_run,
        name="normalize_receipt",
        description="Normalize the most recently extracted raw receipt. Takes no arguments.",
    )


def build_categorize_receipt_tool(
    *, ctx_factory: Callable[[], ToolContext], llm: LLMPort,
    normalized_holder: dict, user_prompt: str | None,
) -> StructuredTool:
    async def _run() -> dict:
        normalized = normalized_holder.get("normalized")
        if normalized is None:
            raise RuntimeError("categorize_receipt called before normalize_receipt")
        result = await categorize_receipt(
            ctx_factory(), llm=llm, normalized=normalized, user_prompt=user_prompt,
        )
        return _dump(result)

    return StructuredTool.from_function(
        coroutine=_run,
        name="categorize_receipt",
        description="Categorize the most recently normalized receipt. Takes no arguments.",
    )


def build_skip_receipt_tool(
    *, ctx_factory: Callable[[], ToolContext], receipt_id_provider: Callable[[], UUID],
) -> StructuredTool:
    from pydantic import BaseModel as _BM, Field as _F

    class _Args(_BM):
        reason: str = _F(..., description="Short human-readable reason for skipping")

    async def _run(reason: str) -> dict:
        result = await skip_receipt(ctx_factory(), receipt_id=receipt_id_provider(), reason=reason)
        return _dump(result)

    return StructuredTool.from_function(
        coroutine=_run,
        name="skip_receipt",
        description="Abandon the current receipt with a short reason. Stops processing this receipt.",
        args_schema=_Args,
    )


def build_aggregate_tool(
    *, ctx_factory: Callable[[], ToolContext], receipts_provider: Callable[[], list[Receipt]],
) -> StructuredTool:
    async def _run() -> dict:
        result = await aggregate_receipts(ctx_factory(), receipts=receipts_provider())
        return _dump(result)

    return StructuredTool.from_function(
        coroutine=_run,
        name="aggregate",
        description="Aggregate totals across all processed receipts. Takes no arguments.",
    )


def build_detect_anomalies_tool(
    *, ctx_factory: Callable[[], ToolContext],
    aggregates_holder: dict, receipts_provider: Callable[[], list[Receipt]],
) -> StructuredTool:
    async def _run() -> list[dict]:
        aggregates = aggregates_holder.get("aggregates")
        if aggregates is None:
            raise RuntimeError("detect_anomalies called before aggregate")
        result = await detect_anomalies(
            ctx_factory(), aggregates=aggregates, receipts=receipts_provider(),
        )
        return _dump(result)

    return StructuredTool.from_function(
        coroutine=_run,
        name="detect_anomalies",
        description="Detect anomalies over the aggregated data. Takes no arguments.",
    )


def build_add_assumption_tool(
    *, ctx_factory: Callable[[], ToolContext], assumptions_sink: list[Issue],
) -> StructuredTool:
    from pydantic import BaseModel as _BM, Field as _F

    class _Args(_BM):
        code: str = _F(..., description="Short kebab/snake-case code")
        message: str = _F(..., description="Human-readable explanation")

    async def _run(code: str, message: str) -> dict:
        result = await add_assumption(ctx_factory(), code=code, message=message)
        assumptions_sink.append(result)
        return _dump(result)

    return StructuredTool.from_function(
        coroutine=_run,
        name="add_assumption",
        description="Add a narrative warning to the final report's issues_and_assumptions.",
        args_schema=_Args,
    )


def build_generate_report_tool(
    *, ctx_factory: Callable[[], ToolContext],
    run_id: UUID,
    aggregates_holder: dict,
    receipts_provider: Callable[[], list[Receipt]],
    issues_provider: Callable[[], list[Issue]],
    report_holder: dict,
    emit_final_result: Callable[[Report], Awaitable[None]],
) -> StructuredTool:
    async def _run() -> dict:
        aggregates = aggregates_holder.get("aggregates")
        if aggregates is None:
            raise RuntimeError("generate_report called before aggregate")
        result = await generate_report(
            ctx_factory(),
            run_id=run_id,
            aggregates=aggregates,
            receipts=receipts_provider(),
            issues=issues_provider(),
        )
        report_holder["report"] = result
        await emit_final_result(result)
        return _dump(result)

    return StructuredTool.from_function(
        coroutine=_run,
        name="generate_report",
        description="Generate the final report and emit final_result. REQUIRED final step.",
    )
