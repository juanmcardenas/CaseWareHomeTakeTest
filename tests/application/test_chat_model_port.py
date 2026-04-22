"""ChatModelPort is an ABC; this smoke-tests importability and abstractness."""
import pytest
from application.ports import ChatModelPort


def test_chat_model_port_is_abstract():
    with pytest.raises(TypeError):
        ChatModelPort()  # type: ignore[abstract]


def test_chat_model_port_requires_build():
    class Incomplete(ChatModelPort):
        pass
    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_deepseek_chat_model_adapter_builds():
    from infrastructure.llm.deepseek_chat_model import DeepSeekChatModelAdapter
    from langchain_core.language_models import BaseChatModel
    adapter = DeepSeekChatModelAdapter(
        api_key="dummy",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        timeout_s=20,
    )
    built = adapter.build()
    assert isinstance(built, BaseChatModel)
