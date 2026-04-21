import pytest
from datetime import datetime, timezone
from uuid import uuid4
from application.events import (
    EventType, RunStarted, Progress, ToolCall, ToolResult,
    ReceiptResult, FinalResult, ErrorEvent, serialize_event,
)


def _now():
    return datetime.now(timezone.utc)


def test_run_started_minimal():
    e = RunStarted(run_id=uuid4(), seq=1, ts=_now(), prompt="conservative")
    assert e.event_type == EventType.RUN_STARTED


def test_progress_with_receipt_scope():
    rid = uuid4()
    rc = uuid4()
    e = Progress(run_id=rid, seq=2, ts=_now(), step="ocr", receipt_id=rc, i=1, n=3)
    assert e.event_type == EventType.PROGRESS
    assert e.receipt_id == rc


def test_tool_call_args_dict():
    e = ToolCall(
        run_id=uuid4(), seq=3, ts=_now(),
        tool="load_images", args={"folder_path": "/tmp"},
    )
    assert e.tool == "load_images"


def test_tool_result_error_flag():
    e = ToolResult(
        run_id=uuid4(), seq=4, ts=_now(),
        tool="extract_receipt_fields", result_summary={"count": 0},
        error=True, duration_ms=120,
    )
    assert e.error is True


def test_receipt_result_status_ok():
    e = ReceiptResult(
        run_id=uuid4(), seq=5, ts=_now(), receipt_id=uuid4(),
        status="ok", vendor="Uber", total="45.67",
        category="Travel", confidence=0.9, issues=[],
    )
    assert e.status == "ok"


def test_final_result_shape():
    e = FinalResult(
        run_id=uuid4(), seq=6, ts=_now(),
        total_spend="100.00", by_category={"Travel": "100.00"},
        receipts=[], issues_and_assumptions=[],
    )
    assert e.event_type == EventType.FINAL_RESULT


def test_error_event_has_code_and_message():
    e = ErrorEvent(run_id=uuid4(), seq=7, ts=_now(), code="no_images", message="0 images")
    assert e.code == "no_images"


def test_serialize_event_is_json_string_with_event_type_field():
    e = RunStarted(run_id=uuid4(), seq=1, ts=_now())
    s = serialize_event(e)
    assert '"event_type"' in s
    assert '"run_started"' in s
