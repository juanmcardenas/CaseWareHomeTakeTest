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
from application.tool_registry import (
    # legacy coroutine tools used by deterministic node methods (Phases 7.4–7.7 replace these)
    aggregate_receipts, categorize_receipt, extract_receipt_fields,
    generate_report, load_images, normalize_receipt,
    # agent-facing tool builders
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
    """Coerce a dict or an ImageRef dataclass to an ImageRef."""
    if isinstance(d, ImageRef):
        return d
    return ImageRef(source_ref=d["source_ref"], local_path=d["local_path"])


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
            reason = agent_error or "agent_did_not_finish"
            receipt = Receipt(
                id=receipt_id,
                source_ref=image.source_ref,
                status="error",
                error=reason,
                issues=[Issue(
                    severity="receipt_error",
                    code="agent_did_not_finish",
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

    async def start(self, state: RunState) -> RunState:
        await self._emit(RunStarted(
            run_id=self.run_id, seq=next(self._seq), ts=_now(), prompt=self.prompt,
        ))
        await self._progress("load_images")
        images = await load_images(self._ctx(), loader=self.image_loader)
        if not images:
            # R4-related: no images is a run-level error (Band C)
            await self._emit(ErrorEvent(
                run_id=self.run_id, seq=next(self._seq), ts=_now(),
                code="no_images", message="no images found in input",
            ))
            return state.model_copy(update={"images": [], "errors": ["no_images"]})
        return state.model_copy(update={"images": images})

    async def process_receipt(self, state: RunState) -> RunState:
        i = state.current
        n = len(state.images)
        image = state.images[i]
        receipt_id = uuid4()

        await self._progress("ocr", receipt_id=receipt_id, i=i + 1, n=n)
        raw: RawReceipt | None = None
        normalized: NormalizedReceipt | None = None
        categorization = None
        receipt_status = "ok"
        err: str | None = None
        local_issues: list[Issue] = []

        try:
            raw = await extract_receipt_fields(self._ctx(receipt_id), ocr=self.ocr, image=image)
        except Exception as e:
            receipt_status = "error"
            err = f"{type(e).__name__}: {e}"
            local_issues.append(Issue(
                severity="receipt_error", code="ocr_failed",
                message=err, receipt_id=receipt_id,
            ))

        if receipt_status == "ok":
            await self._progress("normalize", receipt_id=receipt_id)
            try:
                normalized = await normalize_receipt(self._ctx(receipt_id), raw=raw)  # type: ignore[arg-type]
            except Exception as e:
                receipt_status = "error"
                err = f"{type(e).__name__}: {e}"
                local_issues.append(Issue(
                    severity="receipt_error", code="parse_failed",
                    message=err, receipt_id=receipt_id,
                ))

        if receipt_status == "ok":
            await self._progress("categorize", receipt_id=receipt_id)
            try:
                categorization = await categorize_receipt(
                    self._ctx(receipt_id), llm=self.llm,
                    normalized=normalized,  # type: ignore[arg-type]
                    user_prompt=self.prompt,
                )
            except Exception as e:
                receipt_status = "error"
                err = f"{type(e).__name__}: {e}"
                local_issues.append(Issue(
                    severity="receipt_error", code="llm_failed",
                    message=err, receipt_id=receipt_id,
                ))

        receipt = Receipt(
            id=receipt_id,
            source_ref=image.source_ref,
            vendor=(normalized.vendor if normalized else (raw.vendor if raw else None)),
            receipt_date=normalized.receipt_date if normalized else None,
            receipt_number=(normalized.receipt_number if normalized
                            else (raw.receipt_number if raw else None)),
            total=normalized.total if normalized else None,
            currency=normalized.currency if normalized else None,
            category=categorization.category if categorization else None,
            confidence=categorization.confidence if categorization else None,
            notes=categorization.notes if categorization else None,
            issues=local_issues + (categorization.issues if categorization else []),
            raw_ocr=raw.model_dump(mode="json") if raw else None,
            normalized=normalized.model_dump(mode="json") if normalized else None,
            status=receipt_status,
            error=err,
        )

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
            error_message=err,
        ))

        new_receipts = state.receipts + [receipt]
        new_issues = state.issues + receipt.issues
        return state.model_copy(update={
            "receipts": new_receipts,
            "current": i + 1,
            "issues": new_issues,
        })

    async def finalize(self, state: RunState) -> RunState:
        # R4: if ALL receipts errored, emit run-level error
        if state.receipts and all(r.status != "ok" for r in state.receipts):
            await self._emit(ErrorEvent(
                run_id=self.run_id, seq=next(self._seq), ts=_now(),
                code="all_receipts_failed",
                message=f"all {len(state.receipts)} receipt(s) failed at receipt level",
            ))
            return state

        await self._progress("aggregate")
        agg = await aggregate_receipts(self._ctx(), receipts=state.receipts)
        await self._progress("generate_report")

        # R2: prepend fixed run-level assumptions
        run_level = [
            Issue(severity="warning", code=code, message=msg)
            for code, msg in _RUN_LEVEL_ASSUMPTIONS
        ]
        all_issues = state.issues + run_level

        report = await generate_report(
            self._ctx(),
            run_id=self.run_id,
            aggregates=agg,
            receipts=state.receipts,
            issues=all_issues,
        )
        await self._emit(FinalResult(
            run_id=self.run_id, seq=next(self._seq), ts=_now(),
            total_spend=str(report.total_spend),
            by_category={k: str(v) for k, v in report.by_category.items()},
            receipts=[r.model_dump(mode="json") for r in report.receipts],
            issues_and_assumptions=[iss.model_dump(mode="json") for iss in report.issues_and_assumptions],
        ))
        return state


def build_graph(runner: GraphRunner):
    g = StateGraph(RunState)
    g.add_node("start", runner.start)
    g.add_node("process_receipt", runner.process_receipt)
    g.add_node("finalize", runner.finalize)

    g.add_edge(START, "start")

    def _after_start(state: RunState):
        if state.errors:
            return END
        return "process_receipt" if state.images else "finalize"

    g.add_conditional_edges("start", _after_start, {
        "process_receipt": "process_receipt",
        "finalize": "finalize",
        END: END,
    })

    def _loop_or_finalize(state: RunState):
        return "process_receipt" if state.current < len(state.images) else "finalize"

    g.add_conditional_edges("process_receipt", _loop_or_finalize, {
        "process_receipt": "process_receipt",
        "finalize": "finalize",
    })
    g.add_edge("finalize", END)
    return g.compile()
