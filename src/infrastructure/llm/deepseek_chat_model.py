"""DeepSeek ChatModel adapter for node-level ReAct agents.

DeepSeek's chat API is OpenAI-compatible; we use ChatOpenAI with a custom
base_url. This is the model passed to langchain.agents.create_agent.
"""
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from application.ports import ChatModelPort


class DeepSeekChatModelAdapter(ChatModelPort):
    def __init__(self, *, api_key: str, base_url: str, model: str, timeout_s: int) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._timeout_s = timeout_s

    def build(self) -> BaseChatModel:
        return ChatOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            model=self._model,
            timeout=float(self._timeout_s),
            temperature=0,
        )
