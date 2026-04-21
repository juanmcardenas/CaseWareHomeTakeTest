"""
The 6 tools. Each is a thin wrapper decorated with @traced_tool.

Registry constraint: the application (graph) must only touch receipt data
by calling these. Direct use of adapters or domain functions outside these
wrappers bypasses the trace — don't do that.

Network-sensitive tools (extract_receipt_fields, categorize_receipt) retry
once on network-class exceptions per plan revision R1.
"""
from uuid import UUID
from domain.aggregation import aggregate as _aggregate_pure
from domain.models import (
    Aggregates, AllowedCategory, Categorization, Issue, NormalizedReceipt,
    RawReceipt, Receipt, Report,
)
from domain.normalization import normalize as _normalize_pure
from application.ports import ImageLoaderPort, ImageRef, LLMPort, OCRPort
from application.subagent import categorize_with_subagent
from application.traced_tool import ToolContext, traced_tool


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


TOOL_NAMES = [
    "load_images",
    "extract_receipt_fields",
    "normalize_receipt",
    "categorize_receipt",
    "aggregate",
    "generate_report",
]
