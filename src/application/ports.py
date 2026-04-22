"""
Abstract ports. The application layer depends ONLY on these interfaces.
Infrastructure adapters implement them.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable
from uuid import UUID
from domain.models import Categorization, NormalizedReceipt, RawReceipt
from langchain_core.language_models import BaseChatModel


@dataclass(frozen=True)
class ImageRef:
    """Reference to an image available to the OCR adapter."""
    source_ref: str         # original filename or identifier
    local_path: Path        # absolute path on local disk


class ImageLoaderPort(ABC):
    @abstractmethod
    async def load(self) -> list[ImageRef]: ...


class OCRPort(ABC):
    @abstractmethod
    async def extract(self, image: ImageRef, hint: str | None = None) -> RawReceipt: ...


class LLMPort(ABC):
    """One call per receipt. The `user_prompt` kwarg is accepted for backward
    compatibility with adapter implementations but is currently always passed
    as None — letting the run-level prompt influence categorization was
    observed to collapse the downstream filter into a no-op. Filtering is
    handled by `filter_by_prompt` after categorization."""
    @abstractmethod
    async def categorize(
        self,
        normalized: NormalizedReceipt,
        allowed: list[str],
        user_prompt: str | None,
    ) -> Categorization: ...


class ReportRepositoryPort(ABC):
    @abstractmethod
    async def insert_report(self, row: dict) -> None: ...

    @abstractmethod
    async def update_report(self, report_id: UUID, patch: dict) -> None: ...

    @abstractmethod
    async def insert_receipt(self, row: dict) -> None: ...


class TraceRepositoryPort(ABC):
    @abstractmethod
    async def insert_trace(self, row: dict) -> None: ...


# EventBus subscriber signature
Subscriber = Callable[[dict], Awaitable[None]]


class EventBusPort(ABC):
    @abstractmethod
    async def publish(self, event: dict) -> None: ...

    @abstractmethod
    def subscribe(self, subscriber: Subscriber) -> None: ...


class TracerPort(ABC):
    """Opaque tracer (Langfuse or no-op). Opens a span for a tool call."""
    @abstractmethod
    def start_span(self, name: str, input: dict | None = None) -> "TracerSpan": ...


class TracerSpan(ABC):
    @abstractmethod
    def end(self, output: dict | None = None, error: str | None = None) -> None: ...


class ChatModelPort(ABC):
    """Factory for a LangChain BaseChatModel used by per-node agents.

    Implementations belong in infrastructure/ and must NOT be imported by
    application code outside graph.py.
    """

    @abstractmethod
    def build(self) -> BaseChatModel: ...
