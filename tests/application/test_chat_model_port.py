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
