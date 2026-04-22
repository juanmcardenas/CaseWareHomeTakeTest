"""
The 6 tools. Each is a thin wrapper decorated with @traced_tool.

Registry constraint: the application (graph) must only touch receipt data
by calling these. Direct use of adapters or domain functions outside these
wrappers bypasses the trace — don't do that.

Network-sensitive tools (extract_receipt_fields, categorize_receipt) retry
once on network-class exceptions per plan revision R1.
"""
from decimal import Decimal
from uuid import UUID
from domain.aggregation import aggregate as _aggregate_pure
from domain.models import (
    Aggregates, AllowedCategory, Anomaly, Categorization, Issue, NormalizedReceipt,
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
