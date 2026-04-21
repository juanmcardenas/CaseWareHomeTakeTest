"""Test-facing re-export of the fake chat model implementation."""
from infrastructure.llm.fake_chat_model import (  # noqa: F401
    FakeChatModelAdapter,
    tool_call,
    finish,
    default_mock_script,
)
