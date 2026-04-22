"""POST /runs/stream — multipart OR JSON body; returns SSE."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from application.event_bus import InMemoryEventBus
from application.graph import GraphRunner, RunState, build_graph
from application.ports import ChatModelPort, ReportRepositoryPort, TraceRepositoryPort
from config import LLMMode, Settings
from infrastructure.http.sse import sse_response
from infrastructure.images.folder_loader import LocalFolderImageLoader
from infrastructure.images.list_path_loader import ListPathImageLoader, TooManyPathsError
from infrastructure.images.upload_loader import UploadImageLoader
from infrastructure.ocr.mock_adapter import MockOCRAdapter
from infrastructure.llm.mock_adapter import MockLLMAdapter
from infrastructure.tracing.langfuse_adapter import build_tracer

router = APIRouter()


@dataclass
class RunsDeps:
    settings: Settings
    report_repo: ReportRepositoryPort
    trace_repo: TraceRepositoryPort
    chat_model_port: ChatModelPort


class FolderInput(BaseModel):
    folder_path: str
    prompt: str | None = None


class PathsInput(BaseModel):
    image_paths: list[str]
    prompt: str | None = None


@router.post("/runs/stream")
async def post_runs_stream(request: Request):
    deps: RunsDeps = request.app.state.runs_deps
    settings = deps.settings

    # Branch on Content-Type before reading the body. Using FastAPI's
    # File/Form deps here would eagerly consume the stream as multipart
    # even for JSON requests, breaking a later request.json() call.
    content_type = request.headers.get("content-type", "").lower()
    prompt: str | None = None
    image_loader = None

    if content_type.startswith("multipart/"):
        form = await request.form()
        files: list[UploadFile] = [v for v in form.getlist("files") if isinstance(v, UploadFile)]
        prompt_val = form.get("prompt")
        prompt = prompt_val if isinstance(prompt_val, str) else None
        if not files:
            raise HTTPException(422, "multipart request requires at least one 'files' field")
        if len(files) > settings.max_files_per_run:
            raise HTTPException(413, f"too many files (max {settings.max_files_per_run})")
        image_loader = UploadImageLoader(
            files, settings.allowed_extensions,
            settings.max_file_size_mb * 1024 * 1024,
        )
        input_kind, input_ref = "upload", f"{len(files)} files"
    else:
        body = await request.json()
        has_folder = "folder_path" in body
        has_paths = "image_paths" in body
        if has_folder and has_paths:
            raise HTTPException(422, "provide exactly one of folder_path or image_paths")
        if not has_folder and not has_paths:
            raise HTTPException(422, "body must include folder_path or image_paths")

        if has_paths:
            try:
                pi = PathsInput(**body)
            except Exception as e:
                raise HTTPException(422, f"invalid body: {e}")
            try:
                image_loader = ListPathImageLoader(
                    pi.image_paths,
                    settings.allowed_extensions,
                    settings.assets_dir,
                    max_files_per_run=settings.max_files_per_run,
                )
            except TooManyPathsError as e:
                raise HTTPException(413, str(e))
            except ValueError as e:
                raise HTTPException(400, str(e))
            input_kind, input_ref = "paths", f"{len(pi.image_paths)} paths"
            prompt = pi.prompt
        else:
            try:
                fi = FolderInput(**body)
            except Exception as e:
                raise HTTPException(422, f"invalid body: {e}")
            folder = Path(fi.folder_path).resolve()
            assets = settings.assets_dir.resolve()
            if not str(folder).startswith(str(assets)):
                raise HTTPException(400, f"folder_path must be under {assets}")
            image_loader = LocalFolderImageLoader(folder, settings.allowed_extensions)
            input_kind, input_ref = "folder", str(folder)
            prompt = fi.prompt

    run_id = uuid4()
    bus = InMemoryEventBus()

    async def trace_writer(e: dict) -> None:
        await deps.trace_repo.insert_trace({
            "report_id": run_id,
            "receipt_id": _uuid_or_none(e.get("receipt_id")),
            "seq": e["seq"],
            "event_type": e["event_type"],
            "step": e.get("step"),
            "tool": e.get("tool"),
            "payload": e,
            "duration_ms": e.get("duration_ms"),
            "created_at": datetime.now(timezone.utc),
        })

    bus.subscribe(trace_writer)

    if settings.llm_mode == LLMMode.MOCK:
        ocr = MockOCRAdapter()
        llm = MockLLMAdapter()
    else:
        from infrastructure.ocr.openai_adapter import OpenAIOCRAdapter
        from infrastructure.llm.deepseek_adapter import DeepSeekLLMAdapter
        ocr = OpenAIOCRAdapter(
            api_key=settings.openai_api_key, model=settings.openai_ocr_model,
            timeout_s=settings.ocr_timeout_s,
        )
        llm = DeepSeekLLMAdapter(
            api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url,
            model=settings.deepseek_model, timeout_s=settings.llm_timeout_s,
        )

    tracer = build_tracer(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        run_id=str(run_id),
    )

    await deps.report_repo.insert_report({
        "id": run_id,
        "started_at": datetime.now(timezone.utc),
        "status": "running",
        "prompt": prompt,
        "input_kind": input_kind,
        "input_ref": input_ref,
    })

    runner = GraphRunner(
        run_id=run_id, prompt=prompt,
        bus=bus, tracer=tracer,
        image_loader=image_loader, ocr=ocr, llm=llm,
        chat_model_port=deps.chat_model_port,
        report_repo=deps.report_repo,
    )
    graph = build_graph(runner)

    sse_queue: asyncio.Queue = asyncio.Queue()

    async def sink(e: dict):
        await sse_queue.put(e)

    bus.subscribe(sink)

    async def run_graph_task():
        try:
            await graph.ainvoke(RunState(receipts=[], current=0, errors=[], issues=[]))
            await deps.report_repo.update_report(run_id, {
                "finished_at": datetime.now(timezone.utc),
                "status": "succeeded",
            })
        except Exception as exc:
            import logging
            logging.exception("graph run failed")
            await deps.report_repo.update_report(run_id, {
                "finished_at": datetime.now(timezone.utc),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
        finally:
            await sse_queue.put(None)

    task = asyncio.create_task(run_graph_task())

    async def event_source() -> AsyncIterator[dict]:
        while True:
            e = await sse_queue.get()
            if e is None:
                break
            yield e
        await task

    return sse_response(event_source())


def _uuid_or_none(v) -> UUID | None:
    if v is None:
        return None
    if isinstance(v, UUID):
        return v
    try:
        return UUID(v)
    except Exception:
        return None
