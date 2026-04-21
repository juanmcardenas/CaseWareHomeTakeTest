"""Deterministic LLM mock. Captures the user_prompt it received for assertion."""
from dataclasses import dataclass, field
from domain.models import AllowedCategory, Categorization, NormalizedReceipt, Issue
from application.ports import LLMPort


@dataclass
class MockLLMCall:
    normalized: NormalizedReceipt
    allowed: list[str]
    user_prompt: str | None


class MockLLM(LLMPort):
    def __init__(self,
                 responses: dict[str, Categorization] | None = None,
                 default_category: AllowedCategory = AllowedCategory.OTHER,
                 fail_on: set[str] | None = None) -> None:
        self._responses = responses or {}
        self._default_category = default_category
        self._fail_on = fail_on or set()
        self.calls: list[MockLLMCall] = []

    async def categorize(self, normalized, allowed, user_prompt):
        self.calls.append(MockLLMCall(normalized, list(allowed), user_prompt))
        if normalized.source_ref in self._fail_on:
            raise RuntimeError(f"mock LLM configured to fail on {normalized.source_ref}")
        if normalized.source_ref in self._responses:
            return self._responses[normalized.source_ref]
        return Categorization(
            category=self._default_category,
            confidence=0.8,
            notes="default mock categorization",
            issues=[],
        )
