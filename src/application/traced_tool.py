"""
@traced_tool decorator.

Applies to an async function with signature:
    async def f(ctx: ToolContext, **kwargs) -> <result>

Responsibilities:
- Publish `tool_call` before the call and `tool_result` after — one pair per attempt.
- Measure wall-clock duration (ms) on the `tool_result`.
- On exception: emit `tool_result` with error=true + error_message. If the exception
  matches a retryable type and attempts remain, sleep per backoff schedule and retry
  (emitting a fresh `tool_call` with `attempt=N+1`). Otherwise re-raise.
- Open a tracer span for each attempt.
- Build `result_summary` via a user-supplied `summarize(result)` or default to {}.

NOT responsible for:
- Converting exceptions into recoverable Receipt errors (that's done by the graph).
"""
from __future__ import annotations
import asyncio
import functools
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterator
from uuid import UUID

from application.events import ToolCall, ToolResult
from application.ports import EventBusPort, TracerPort


_NETWORK_EXCEPTIONS: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError, OSError, ConnectionError,
)


@dataclass
class ToolContext:
    run_id: UUID
    bus: EventBusPort
    tracer: TracerPort
    seq_counter: Iterator[int]
    receipt_id: UUID | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def traced_tool(
    tool_name: str,
    summarize: Callable[[Any], dict] | None = None,
    retries: int = 0,
    retry_delays_s: tuple[float, ...] = (1.0, 2.0),
    retry_on: tuple[type[BaseException], ...] = _NETWORK_EXCEPTIONS,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    def decorator(fn: Callable[..., Awaitable[Any]]):
        @functools.wraps(fn)
        async def wrapper(ctx: ToolContext, /, **kwargs) -> Any:
            attempt = 1
            while True:
                call_event = ToolCall(
                    run_id=ctx.run_id,
                    seq=next(ctx.seq_counter),
                    ts=_now(),
                    tool=tool_name,
                    receipt_id=ctx.receipt_id,
                    attempt=attempt,
                    args=_safe_args(kwargs),
                )
                await ctx.bus.publish(call_event.model_dump(mode="json"))
                span = ctx.tracer.start_span(tool_name, input=_safe_args(kwargs))
                started = time.perf_counter()
                try:
                    result = await fn(ctx, **kwargs)
                except Exception as exc:
                    duration_ms = int((time.perf_counter() - started) * 1000)
                    err_event = ToolResult(
                        run_id=ctx.run_id,
                        seq=next(ctx.seq_counter),
                        ts=_now(),
                        tool=tool_name,
                        receipt_id=ctx.receipt_id,
                        result_summary={},
                        error=True,
                        error_message=f"{type(exc).__name__}: {exc}",
                        duration_ms=duration_ms,
                    )
                    await ctx.bus.publish(err_event.model_dump(mode="json"))
                    span.end(error=str(exc))
                    if attempt <= retries and isinstance(exc, retry_on):
                        delay = retry_delays_s[min(attempt - 1, len(retry_delays_s) - 1)]
                        await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    raise
                duration_ms = int((time.perf_counter() - started) * 1000)
                summary = (summarize(result) if summarize else {}) or {}
                ok_event = ToolResult(
                    run_id=ctx.run_id,
                    seq=next(ctx.seq_counter),
                    ts=_now(),
                    tool=tool_name,
                    receipt_id=ctx.receipt_id,
                    result_summary=summary,
                    error=False,
                    duration_ms=duration_ms,
                )
                await ctx.bus.publish(ok_event.model_dump(mode="json"))
                span.end(output=summary)
                return result

        return wrapper

    return decorator


def _safe_args(kwargs: dict) -> dict:
    """Best-effort JSON-safe rendering of kwargs for the SSE payload."""
    out: dict = {}
    for k, v in kwargs.items():
        try:
            if hasattr(v, "model_dump"):
                out[k] = v.model_dump(mode="json")
            elif isinstance(v, (str, int, float, bool, type(None), list, dict)):
                out[k] = v
            else:
                out[k] = str(v)
        except Exception:
            out[k] = f"<unserializable {type(v).__name__}>"
    return out
