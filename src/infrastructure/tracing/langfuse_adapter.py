"""
Langfuse tracer.

If `public_key`/`secret_key` are empty, returns a no-op tracer. This keeps
Langfuse optional for reviewers who don't set up credentials.

The API uses manual trace+span lifecycle; we treat each tool call as a span
under a single trace opened per run (constructed by `build_tracer`).
"""
from __future__ import annotations
import logging
from typing import Any
from application.ports import TracerPort, TracerSpan

_log = logging.getLogger(__name__)


class NoopSpan(TracerSpan):
    def end(self, output=None, error=None):
        pass


class NoopTracer(TracerPort):
    def start_span(self, name, input=None):
        return NoopSpan()


class LangfuseTracer(TracerPort):
    def __init__(self, *, public_key: str, secret_key: str, host: str, run_id: str) -> None:
        from langfuse import Langfuse
        self._client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        self._trace = self._client.trace(name="receipt_run", id=run_id)

    def start_span(self, name: str, input: dict | None = None) -> "LangfuseSpan":
        span = self._trace.span(name=name, input=input or {})
        return LangfuseSpan(span)


class LangfuseSpan(TracerSpan):
    def __init__(self, span: Any) -> None:
        self._span = span

    def end(self, output: dict | None = None, error: str | None = None) -> None:
        try:
            if error:
                self._span.end(level="ERROR", status_message=error)
            else:
                self._span.end(output=output or {})
        except Exception:
            _log.exception("langfuse span end failed (ignored)")


def build_tracer(*, public_key: str | None, secret_key: str | None, host: str, run_id: str) -> TracerPort:
    if not public_key or not secret_key:
        return NoopTracer()
    try:
        return LangfuseTracer(public_key=public_key, secret_key=secret_key, host=host, run_id=run_id)
    except Exception:
        _log.exception("Langfuse init failed; falling back to no-op tracer")
        return NoopTracer()
