"""FakeChatModelAdapter + DSL helpers for scripting agent tests.

`tool_call(name, args)` builds an AIMessage that create_agent interprets as
"call tool X with these args". `finish(content)` builds an AIMessage with no
tool calls, which ends the ReAct loop. `FakeChatModelAdapter(script)` wraps
`FakeMessagesListChatModel` so a scripted sequence can drive one or more
create_agent subgraphs deterministically.
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


class FakeChatModelAdapter(ChatModelPort):
    def __init__(self, script: Iterable[AIMessage]) -> None:
        self._script = list(script)

    def build(self) -> BaseChatModel:
        return FakeMessagesListChatModel(responses=self._script)
