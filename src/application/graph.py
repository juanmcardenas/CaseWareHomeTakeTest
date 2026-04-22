"""
LangGraph state machine: three per-node ReAct agents + deterministic edges.

See docs/superpowers/specs/2026-04-21-agentic-graph-design.md.

Graph:
    START -> ingest_node -> (cond) ─┬─▶ END (run-level error)
                                    └─▶ per_receipt_node ⇄ (cond loop) ⇄ finalize_node -> END

Per-node agents are compiled via langchain.agents.create_agent and invoked
by thin wrappers that project typed RunState in/out.
"""
from __future__ import annotations
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from domain.models import (
    AllowedCategory, Anomaly, Issue, NormalizedReceipt, RawReceipt, Receipt, Aggregates,
)
from application.events import (
    ErrorEvent, FinalResult, Progress, ReceiptResult, RunStarted,
)
from application.ports import (
    ChatModelPort, EventBusPort, ImageLoaderPort, ImageRef, LLMPort, OCRPort,
    ReportRepositoryPort, TracerPort,
)
from application.traced_tool import ToolContext
from application.agent_prompts import (
    INGEST_SYSTEM_PROMPT, PER_RECEIPT_SYSTEM_PROMPT, FINALIZE_SYSTEM_PROMPT,
)
from infrastructure.agent_tools import (
    build_load_images_tool, build_filter_by_prompt_tool,
    build_extract_receipt_fields_tool, build_re_extract_with_hint_tool,
    build_normalize_receipt_tool, build_categorize_receipt_tool,
    build_skip_receipt_tool, build_aggregate_tool, build_detect_anomalies_tool,
    build_add_assumption_tool, build_generate_report_tool,
)


# Fixed run-level assumptions appended to every run's issues_and_assumptions
_RUN_LEVEL_ASSUMPTIONS: list[tuple[str, str]] = [
    ("only_allowed_extensions", "Only files matching jpg/jpeg/png/webp were considered."),
    ("default_currency_usd", "Totals assume USD when currency is absent."),
    ("errored_receipts_excluded", "Receipts with OCR/normalization/LLM failures are excluded from aggregation."),
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_image_ref(d) -> "ImageRef":
    """Coerce a dict or an ImageRef dataclass to an ImageRef.

    When an ImageRef flows through Pydantic's model_dump(mode="json")
    (e.g. as part of FilterResult in filter_by_prompt's return value),
    local_path is serialized to a string. ImageRef is a frozen dataclass
    that does not coerce types on construction, so we must explicitly
    wrap the path back into a Path — otherwise downstream consumers call
    image.local_path.read_bytes() on a string.
    """
    if isinstance(d, ImageRef):
        return d
    return ImageRef(source_ref=d["source_ref"], local_path=Path(d["local_path"]))


def _capture_list(tool, sink: list, item_cls):
    """Wrap a StructuredTool so its list result is captured into `sink`."""
    from langchain_core.tools import StructuredTool

    original_coro = tool.coroutine

    async def _wrapped(*args, **kwargs):
        result = await original_coro(*args, **kwargs)
        sink.clear()
        # result may be list[dict] or list[ImageRef] depending on _dump behaviour
        if item_cls is ImageRef:
            for d in result:
                sink.append(_to_image_ref(d))
        else:
            sink.extend(item_cls(**d) for d in result)
        return result

    return StructuredTool.from_function(
        coroutine=_wrapped,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def _capture_filter(tool, images_sink: list, dropped_sink: list):
    """Wrap the filter_by_prompt StructuredTool so kept/dropped lists are captured."""
    from langchain_core.tools import StructuredTool

    original_coro = tool.coroutine

    async def _wrapped(*args, **kwargs):
        result = await original_coro(*args, **kwargs)
        # result is a dict: {"kept": [...ImageRef-like...], "dropped": [...tuples/lists...]}
        images_sink.clear()
        for d in result["kept"]:
            images_sink.append(_to_image_ref(d))
        dropped_sink.clear()
        for d in result["dropped"]:
            dropped_sink.append(tuple(d))
        return result

    return StructuredTool.from_function(
        coroutine=_wrapped,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def _capture_raw(tool, holder: dict):
    """Capture the RawReceipt output of extract_receipt_fields / re_extract_with_hint."""
    from langchain_core.tools import StructuredTool
    original_coro = tool.coroutine

    async def _wrapped(*args, **kwargs):
        result = await original_coro(*args, **kwargs)
        # result is a dict from _dump(RawReceipt)
        holder["raw"] = RawReceipt(**result) if isinstance(result, dict) else result
        return result

    return StructuredTool.from_function(
        coroutine=_wrapped, name=tool.name, description=tool.description,
        args_schema=tool.args_schema,
    )


def _capture_normalized(tool, holder: dict):
    """Capture the NormalizedReceipt output of normalize_receipt."""
    from langchain_core.tools import StructuredTool
    original_coro = tool.coroutine

    async def _wrapped(*args, **kwargs):
        result = await original_coro(*args, **kwargs)
        holder["normalized"] = NormalizedReceipt(**result) if isinstance(result, dict) else result
        return result

    return StructuredTool.from_function(
        coroutine=_wrapped, name=tool.name, description=tool.description,
        args_schema=tool.args_schema,
    )


def _capture_categorization(tool, holder: dict):
    """Capture the Categorization output of categorize_receipt."""
    from langchain_core.tools import StructuredTool
    from domain.models import Categorization
    original_coro = tool.coroutine

    async def _wrapped(*args, **kwargs):
        result = await original_coro(*args, **kwargs)
        holder["categorization"] = Categorization(**result) if isinstance(result, dict) else result
        return result

    return StructuredTool.from_function(
        coroutine=_wrapped, name=tool.name, description=tool.description,
        args_schema=tool.args_schema,
    )


def _capture_receipt(tool, holder: dict):
    """Capture the Receipt output of skip_receipt."""
    from langchain_core.tools import StructuredTool
    original_coro = tool.coroutine

    async def _wrapped(*args, **kwargs):
        result = await original_coro(*args, **kwargs)
        holder["receipt"] = Receipt(**result) if isinstance(result, dict) else result
        return result

    return StructuredTool.from_function(
        coroutine=_wrapped, name=tool.name, description=tool.description,
        args_schema=tool.args_schema,
    )


def _capture_aggregates(tool, holder: dict):
    """Capture the Aggregates output of the aggregate tool."""
    from langchain_core.tools import StructuredTool
    original_coro = tool.coroutine

    async def _wrapped(*args, **kwargs):
        result = await original_coro(*args, **kwargs)
        # result is a dict from _dump(Aggregates); Decimal values serialized to strings
        if isinstance(result, dict):
            holder["aggregates"] = Aggregates(**result)
        else:
            holder["aggregates"] = result
        return result

    return StructuredTool.from_function(
        coroutine=_wrapped, name=tool.name, description=tool.description,
        args_schema=tool.args_schema,
    )


class RunState(BaseModel):
    images: list[ImageRef] = Field(default_factory=list)
    filtered_out: list[tuple[str, str]] = Field(default_factory=list)
    receipts: list[Receipt] = Field(default_factory=list)
    current: int = 0
    errors: list[str] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    assumptions_added_by_agent: list[Issue] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class GraphRunner:
    def __init__(
        self, *,
        run_id: UUID,
        prompt: str | None,
        bus: EventBusPort,
        tracer: TracerPort,
        image_loader: ImageLoaderPort,
        ocr: OCRPort,
        llm: LLMPort,
        chat_model_port: ChatModelPort,
        report_repo: ReportRepositoryPort,
    ) -> None:
        self.run_id = run_id
        self.prompt = prompt
        self.bus = bus
        self.tracer = tracer
        self.image_loader = image_loader
        self.ocr = ocr
        self.llm = llm
        self.chat_model_port = chat_model_port
        self.report_repo = report_repo
        self._seq = count(1)

    def _ctx(self, receipt_id: UUID | None = None) -> ToolContext:
        return ToolContext(
            run_id=self.run_id, bus=self.bus, tracer=self.tracer,
            seq_counter=self._seq, receipt_id=receipt_id,
        )

    async def _emit(self, event_model) -> None:
        await self.bus.publish(event_model.model_dump(mode="json"))

    async def _progress(self, step: str, receipt_id: UUID | None = None,
                        i: int | None = None, n: int | None = None) -> None:
        await self._emit(Progress(
            run_id=self.run_id, seq=next(self._seq), ts=_now(),
            step=step, receipt_id=receipt_id, i=i, n=n,
        ))

    async def ingest_node(self, state: RunState) -> RunState:
        # Pre-node: emit run_started and a progress marker for ingest
        await self._emit(RunStarted(
            run_id=self.run_id, seq=next(self._seq), ts=_now(), prompt=self.prompt,
        ))
        await self._progress("ingest_start")

        # Holders for capturing tool outputs
        images_holder: list[ImageRef] = []
        dropped_holder: list[tuple[str, str]] = []

        # Construct the two tools
        load_tool = build_load_images_tool(
            ctx_factory=lambda: self._ctx(),
            loader=self.image_loader,
        )
        filter_tool = build_filter_by_prompt_tool(
            ctx_factory=lambda: self._ctx(),
            images_provider=lambda: list(images_holder),
            user_prompt=self.prompt,
        )

        # Wrap so their outputs are captured into the holders
        wrapped_load = _capture_list(load_tool, images_holder, ImageRef)
        wrapped_filter = _capture_filter(filter_tool, images_holder, dropped_holder)

        # Build the subgraph agent
        agent = create_agent(
            model=self.chat_model_port.build(),
            tools=[wrapped_load, wrapped_filter],
            system_prompt=INGEST_SYSTEM_PROMPT,
        )

        human = HumanMessage(content=self.prompt or "Process all receipts.")
        # Iteration bound: LangGraph's default recursion_limit (25) caps ReAct loops
        # in the outer graph. For the inner agent built by create_agent, the default
        # is 9999; in practice, a looping model hits a routing KeyError before that
        # limit. Either way, any Exception (GraphRecursionError, KeyError, etc.) is
        # caught below and converted into a run-level ingest_iterations_exhausted
        # event. Same mechanism applies in finalize_node.
        try:
            await agent.ainvoke({"messages": [human]})
        except Exception as e:
            await self._emit(ErrorEvent(
                run_id=self.run_id, seq=next(self._seq), ts=_now(),
                code="ingest_iterations_exhausted",
                message=f"ingest_node iteration cap or error: {type(e).__name__}: {e}",
            ))
            return state.model_copy(update={"errors": state.errors + ["ingest_iterations_exhausted"]})

        await self._progress("ingest_done", n=len(images_holder))
        return state.model_copy(update={
            "images": list(images_holder),
            "filtered_out": list(dropped_holder),
        })

    async def per_receipt_node(self, state: RunState) -> RunState:
        i = state.current
        n = len(state.images)
        image = state.images[i]
        receipt_id = uuid4()

        await self._progress("process_receipt", receipt_id=receipt_id, i=i + 1, n=n)

        raw_holder: dict = {}
        normalized_holder: dict = {}
        categorization_holder: dict = {}
        skip_holder: dict = {}

        extract_tool = _capture_raw(
            build_extract_receipt_fields_tool(
                ctx_factory=lambda: self._ctx(receipt_id),
                ocr=self.ocr, image_provider=lambda: image,
            ),
            raw_holder,
        )
        reextract_tool = _capture_raw(
            build_re_extract_with_hint_tool(
                ctx_factory=lambda: self._ctx(receipt_id),
                ocr=self.ocr, image_provider=lambda: image,
            ),
            raw_holder,
        )
        normalize_tool = _capture_normalized(
            build_normalize_receipt_tool(
                ctx_factory=lambda: self._ctx(receipt_id),
                raw_holder=raw_holder,
            ),
            normalized_holder,
        )
        categorize_tool = _capture_categorization(
            build_categorize_receipt_tool(
                ctx_factory=lambda: self._ctx(receipt_id),
                llm=self.llm, normalized_holder=normalized_holder,
                user_prompt=self.prompt,
            ),
            categorization_holder,
        )
        skip_tool = _capture_receipt(
            build_skip_receipt_tool(
                ctx_factory=lambda: self._ctx(receipt_id),
                receipt_id_provider=lambda: receipt_id,
            ),
            skip_holder,
        )

        agent = create_agent(
            model=self.chat_model_port.build(),
            tools=[extract_tool, reextract_tool, normalize_tool, categorize_tool, skip_tool],
            system_prompt=PER_RECEIPT_SYSTEM_PROMPT,
        )

        human = HumanMessage(content=(
            f"Process receipt index {i+1}/{n}: source_ref={image.source_ref}, receipt_id={receipt_id}"
        ))

        agent_error: str | None = None
        try:
            await agent.ainvoke({"messages": [human]})
        except Exception as e:
            agent_error = f"{type(e).__name__}: {e}"

        # Assemble the Receipt
        if skip_holder.get("receipt") is not None:
            receipt = skip_holder["receipt"]
            # Overwrite the empty source_ref from skip_receipt with the actual image
            receipt = receipt.model_copy(update={"source_ref": image.source_ref})
        elif categorization_holder.get("categorization") is not None:
            raw = raw_holder.get("raw")
            normalized = normalized_holder.get("normalized")
            cat = categorization_holder["categorization"]
            receipt = Receipt(
                id=receipt_id,
                source_ref=image.source_ref,
                vendor=(normalized.vendor if normalized else (raw.vendor if raw else None)),
                receipt_date=normalized.receipt_date if normalized else None,
                receipt_number=(normalized.receipt_number if normalized
                                else (raw.receipt_number if raw else None)),
                total=normalized.total if normalized else None,
                currency=normalized.currency if normalized else None,
                category=cat.category,
                confidence=cat.confidence,
                notes=cat.notes,
                issues=list(cat.issues),
                raw_ocr=raw.model_dump(mode="json") if raw else None,
                normalized=normalized.model_dump(mode="json") if normalized else None,
                status="ok",
            )
        else:
            # If a tool raised, agent_error is set and carries the real cause
            # (e.g., "ValueError: unparseable date: '...'"). Otherwise the agent
            # simply stopped without producing a receipt.
            if agent_error:
                reason = agent_error
                code = "tool_failed"
            else:
                reason = "agent_did_not_finish"
                code = "agent_did_not_finish"
            receipt = Receipt(
                id=receipt_id,
                source_ref=image.source_ref,
                status="error",
                error=reason,
                issues=[Issue(
                    severity="receipt_error",
                    code=code,
                    message=reason,
                    receipt_id=receipt_id,
                )],
            )

        # Persist
        await self.report_repo.insert_receipt({
            "id": receipt.id, "report_id": self.run_id, "seq": i + 1,
            "source_ref": receipt.source_ref, "vendor": receipt.vendor,
            "receipt_date": receipt.receipt_date, "receipt_number": receipt.receipt_number,
            "total": receipt.total, "currency": receipt.currency,
            "category": receipt.category.value if receipt.category else None,
            "confidence": receipt.confidence, "notes": receipt.notes,
            "issues": [iss.model_dump(mode="json") for iss in receipt.issues],
            "raw_ocr": receipt.raw_ocr, "normalized": receipt.normalized,
            "status": receipt.status, "error": receipt.error,
            "created_at": _now(),
        })
        # Emit
        await self._emit(ReceiptResult(
            run_id=self.run_id, seq=next(self._seq), ts=_now(),
            receipt_id=receipt.id, status=receipt.status,
            vendor=receipt.vendor,
            receipt_date=receipt.receipt_date.isoformat() if receipt.receipt_date else None,
            receipt_number=receipt.receipt_number,
            total=str(receipt.total) if receipt.total is not None else None,
            currency=receipt.currency,
            category=receipt.category.value if receipt.category else None,
            confidence=receipt.confidence,
            notes=receipt.notes,
            issues=[iss.model_dump(mode="json") for iss in receipt.issues],
            error_message=receipt.error,
        ))

        return state.model_copy(update={
            "receipts": state.receipts + [receipt],
            "current": i + 1,
            "issues": state.issues + list(receipt.issues),
        })

    async def finalize_node(self, state: RunState) -> RunState:
        # R4 short-circuit — deterministic, agent NOT invoked
        if state.receipts and all(r.status != "ok" for r in state.receipts):
            await self._emit(ErrorEvent(
                run_id=self.run_id, seq=next(self._seq), ts=_now(),
                code="all_receipts_failed",
                message=f"all {len(state.receipts)} receipt(s) failed at receipt level",
            ))
            return state

        await self._progress("finalize_start")

        aggregates_holder: dict = {}
        report_holder: dict = {}
        assumptions_sink: list[Issue] = []

        aggregate_tool = _capture_aggregates(
            build_aggregate_tool(
                ctx_factory=lambda: self._ctx(),
                receipts_provider=lambda: list(state.receipts),
            ),
            aggregates_holder,
        )

        detect_tool = build_detect_anomalies_tool(
            ctx_factory=lambda: self._ctx(),
            aggregates_holder=aggregates_holder,
            receipts_provider=lambda: list(state.receipts),
        )

        add_assumption_tool = build_add_assumption_tool(
            ctx_factory=lambda: self._ctx(),
            assumptions_sink=assumptions_sink,
        )

        def _issues_provider() -> list[Issue]:
            run_level = [
                Issue(severity="warning", code=code, message=msg)
                for code, msg in _RUN_LEVEL_ASSUMPTIONS
            ]
            return state.issues + run_level + list(assumptions_sink)

        async def _emit_final(report):
            await self._emit(FinalResult(
                run_id=self.run_id, seq=next(self._seq), ts=_now(),
                total_spend=str(report.total_spend),
                by_category={k: str(v) for k, v in report.by_category.items()},
                receipts=[r.model_dump(mode="json") for r in report.receipts],
                issues_and_assumptions=[iss.model_dump(mode="json") for iss in report.issues_and_assumptions],
            ))

        generate_report_tool = build_generate_report_tool(
            ctx_factory=lambda: self._ctx(),
            run_id=self.run_id,
            aggregates_holder=aggregates_holder,
            receipts_provider=lambda: list(state.receipts),
            issues_provider=_issues_provider,
            report_holder=report_holder,
            emit_final_result=_emit_final,
        )

        agent = create_agent(
            model=self.chat_model_port.build(),
            tools=[aggregate_tool, detect_tool, add_assumption_tool, generate_report_tool],
            system_prompt=FINALIZE_SYSTEM_PROMPT,
        )

        human = HumanMessage(content="Produce the final report.")

        try:
            await agent.ainvoke({"messages": [human]})
        except Exception as e:
            await self._emit(ErrorEvent(
                run_id=self.run_id, seq=next(self._seq), ts=_now(),
                code="finalize_iterations_exhausted",
                message=f"finalize_node failed: {type(e).__name__}: {e}",
            ))
            return state

        if report_holder.get("report") is None:
            await self._emit(ErrorEvent(
                run_id=self.run_id, seq=next(self._seq), ts=_now(),
                code="no_final_report",
                message="finalize agent finished without calling generate_report",
            ))
            return state

        return state.model_copy(update={
            "assumptions_added_by_agent": list(assumptions_sink),
        })

def build_graph(runner: "GraphRunner"):
    g = StateGraph(RunState)
    g.add_node("ingest_node", runner.ingest_node)
    g.add_node("per_receipt_node", runner.per_receipt_node)
    g.add_node("finalize_node", runner.finalize_node)

    g.add_edge(START, "ingest_node")

    async def _after_ingest(state: RunState):
        # If ingest errored (e.g. iterations exhausted), the wrapper already emitted
        # the ErrorEvent and set state.errors; terminate the run here.
        if state.errors:
            return END
        if len(state.images) == 0 and len(state.filtered_out) == 0:
            await runner._emit(ErrorEvent(
                run_id=runner.run_id, seq=next(runner._seq), ts=_now(),
                code="no_images", message="no images found in input",
            ))
            return END
        if len(state.images) == 0 and len(state.filtered_out) > 0:
            await runner._emit(ErrorEvent(
                run_id=runner.run_id, seq=next(runner._seq), ts=_now(),
                code="all_images_filtered_out",
                message=f"all {len(state.filtered_out)} image(s) were filtered out by prompt",
            ))
            return END
        return "per_receipt_node"

    g.add_conditional_edges("ingest_node", _after_ingest, {
        "per_receipt_node": "per_receipt_node",
        END: END,
    })

    def _loop_or_finalize(state: RunState):
        return "per_receipt_node" if state.current < len(state.images) else "finalize_node"

    g.add_conditional_edges("per_receipt_node", _loop_or_finalize, {
        "per_receipt_node": "per_receipt_node",
        "finalize_node": "finalize_node",
    })
    g.add_edge("finalize_node", END)
    return g.compile()
