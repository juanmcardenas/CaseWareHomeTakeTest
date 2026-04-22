"""Scripted ChatModel adapter used in mock mode.

When LLM_MODE=mock, composition_root uses FakeChatModelAdapter with
default_mock_script(...) to drive the agentic graph end-to-end without
real API keys. Also reused by tests (re-exported from tests/fakes/).
"""
from __future__ import annotations
from typing import Iterable
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from application.ports import ChatModelPort


def tool_call(name: str, args: dict, id: str | None = None) -> AIMessage:
    """Build an AIMessage that invokes a single tool."""
    return AIMessage(
        content="",
        tool_calls=[{
            "name": name,
            "args": args,
            "id": id or f"tc-{uuid4().hex[:8]}",
            "type": "tool_call",
        }],
    )


def finish(content: str = "") -> AIMessage:
    """Build an AIMessage with no tool calls (terminates the ReAct loop)."""
    return AIMessage(content=content)


class _ToolAwareFakeModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel extended with a no-op bind_tools.

    create_agent calls bind_tools on the model at construction time. The fake
    model raises NotImplementedError by default; this subclass returns self so
    the scripted AIMessages (which already carry tool_calls) are used as-is.
    """

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self


class FakeChatModelAdapter(ChatModelPort):
    def __init__(self, script: Iterable[AIMessage]) -> None:
        self._model = _ToolAwareFakeModel(responses=list(script))

    def build(self) -> BaseChatModel:
        # Always return the same model instance so the script position is
        # preserved across multiple node invocations within one run.
        return self._model


def default_mock_script(max_receipts: int = 25) -> list[AIMessage]:
    """Ship a script long enough for a typical mock run with up to max_receipts images.

    Sequence: ingest (load_images + finish), then per-receipt
    (extract, normalize, categorize, finish) × max_receipts, then finalize
    (aggregate, detect_anomalies, generate_report, finish).
    """
    out: list[AIMessage] = []
    # Ingest
    out.append(tool_call("load_images", {}))
    out.append(finish())
    # Per-receipt × max_receipts
    for _ in range(max_receipts):
        out.append(tool_call("extract_receipt_fields", {}))
        out.append(tool_call("normalize_receipt", {}))
        out.append(tool_call("categorize_receipt", {}))
        out.append(finish())
    # Finalize
    out.append(tool_call("aggregate", {}))
    out.append(tool_call("detect_anomalies", {}))
    out.append(tool_call("generate_report", {}))
    out.append(finish())
    return out
