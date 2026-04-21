from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Literal, Union
from uuid import UUID
from pydantic import BaseModel, Field


class EventType(str, Enum):
    RUN_STARTED = "run_started"
    PROGRESS = "progress"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RECEIPT_RESULT = "receipt_result"
    FINAL_RESULT = "final_result"
    ERROR = "error"


class _EventBase(BaseModel):
    run_id: UUID
    seq: int
    ts: datetime


class RunStarted(_EventBase):
    event_type: Literal[EventType.RUN_STARTED] = EventType.RUN_STARTED
    prompt: str | None = None
    receipt_count_estimate: int | None = None


class Progress(_EventBase):
    event_type: Literal[EventType.PROGRESS] = EventType.PROGRESS
    step: str
    receipt_id: UUID | None = None
    i: int | None = None
    n: int | None = None


class ToolCall(_EventBase):
    event_type: Literal[EventType.TOOL_CALL] = EventType.TOOL_CALL
    tool: str
    receipt_id: UUID | None = None
    attempt: int = 1
    args: dict = Field(default_factory=dict)


class ToolResult(_EventBase):
    event_type: Literal[EventType.TOOL_RESULT] = EventType.TOOL_RESULT
    tool: str
    receipt_id: UUID | None = None
    result_summary: dict = Field(default_factory=dict)
    error: bool = False
    error_message: str | None = None
    duration_ms: int | None = None


class ReceiptResult(_EventBase):
    event_type: Literal[EventType.RECEIPT_RESULT] = EventType.RECEIPT_RESULT
    receipt_id: UUID
    status: Literal["ok", "error"]
    vendor: str | None = None
    receipt_date: str | None = None        # ISO; kept as string in the wire event
    receipt_number: str | None = None
    total: str | None = None               # Decimal -> str for wire
    currency: str | None = None
    category: str | None = None
    confidence: float | None = None
    notes: str | None = None
    issues: list[dict] = Field(default_factory=list)
    error_message: str | None = None


class FinalResult(_EventBase):
    event_type: Literal[EventType.FINAL_RESULT] = EventType.FINAL_RESULT
    total_spend: str
    by_category: dict[str, str]
    receipts: list[dict]
    issues_and_assumptions: list[dict]


class ErrorEvent(_EventBase):
    event_type: Literal[EventType.ERROR] = EventType.ERROR
    code: str
    message: str


Event = Union[
    RunStarted, Progress, ToolCall, ToolResult,
    ReceiptResult, FinalResult, ErrorEvent,
]


def serialize_event(e: Event) -> str:
    """Serialize an event to a JSON string suitable for an SSE `data:` field."""
    return e.model_dump_json()
