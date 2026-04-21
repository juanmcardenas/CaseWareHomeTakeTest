"""Exercises the FakeChatModelAdapter + DSL helpers used by agent tests."""
from uuid import uuid4
import pytest

from tests.fakes.fake_chat_model import (
    FakeChatModelAdapter, finish, tool_call,
)


def test_finish_produces_ai_message_with_no_tool_calls():
    msg = finish("done")
    assert msg.content == "done"
    assert msg.tool_calls == []


def test_tool_call_produces_ai_message_with_one_tool_call():
    msg = tool_call("extract_receipt_fields", {"image": "x"})
    assert msg.tool_calls[0]["name"] == "extract_receipt_fields"
    assert msg.tool_calls[0]["args"] == {"image": "x"}


def test_fake_chat_model_adapter_builds_a_chat_model_that_replays_script():
    script = [tool_call("foo", {"x": 1}), finish("bye")]
    adapter = FakeChatModelAdapter(script)
    built = adapter.build()
    out1 = built.invoke("anything")
    out2 = built.invoke("anything")
    assert out1.tool_calls[0]["name"] == "foo"
    assert out2.content == "bye"
