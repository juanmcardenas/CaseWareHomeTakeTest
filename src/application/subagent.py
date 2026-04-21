"""
Categorization sub-agent.

One LLM call per receipt. Sub-agent wires:
  - normalized fields (type-safe)
  - the list of allowed category strings
  - the user's optional prompt (injected into the LLMPort implementation's system message)
and returns a validated Categorization.

The actual prompt construction and provider call live in the LLM adapter.
This module only coordinates the call and re-raises on failure.
"""
from domain.models import AllowedCategory, Categorization, NormalizedReceipt
from application.ports import LLMPort


ALLOWED_CATEGORY_VALUES: list[str] = [c.value for c in AllowedCategory]


async def categorize_with_subagent(
    llm: LLMPort,
    normalized: NormalizedReceipt,
    user_prompt: str | None,
) -> Categorization:
    return await llm.categorize(
        normalized=normalized,
        allowed=ALLOWED_CATEGORY_VALUES,
        user_prompt=user_prompt,
    )
