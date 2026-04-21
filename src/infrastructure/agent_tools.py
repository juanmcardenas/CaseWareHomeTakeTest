"""Agent-facing LangChain tool builders.

These wrap the @traced_tool functions in src/application/tool_registry.py as
LangChain StructuredTool objects usable by langchain.agents.create_agent.

Kept in infrastructure/ so the application layer never imports LangChain
types — per the hexagonal contract established in ports.py.
"""
from __future__ import annotations
from typing import Awaitable, Callable
from uuid import UUID

from langchain_core.tools import StructuredTool
from pydantic import BaseModel as _BM, Field as _F

from application.ports import ImageLoaderPort, ImageRef, LLMPort, OCRPort
from application.traced_tool import ToolContext
from application.tool_registry import (
    load_images, filter_by_prompt, extract_receipt_fields,
    re_extract_with_hint, normalize_receipt, categorize_receipt,
    skip_receipt, aggregate_receipts, detect_anomalies,
    add_assumption, generate_report,
)
from domain.models import Issue, Receipt, Report


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
