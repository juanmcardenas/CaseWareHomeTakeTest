"""
Categorization sub-agent.

One LLM call per receipt. Sub-agent wires:
  - normalized fields (type-safe)
  - the list of allowed category strings
and returns a validated Categorization.

The run-level user prompt is intentionally NOT passed here — filtering
is a separate concern handled by filter_by_prompt after categorization.
Letting the prompt influence categorization was observed to collapse
the filter into a no-op (the LLM would re-label receipts to match the
user's "only X" intent).

The actual prompt construction and provider call live in the LLM adapter.
This module only coordinates the call and re-raises on failure.
"""
from domain.models import AllowedCategory, Categorization, NormalizedReceipt
from application.ports import LLMPort


ALLOWED_CATEGORY_VALUES: list[str] = [c.value for c in AllowedCategory]


async def categorize_with_subagent(
    llm: LLMPort,
    normalized: NormalizedReceipt,
) -> Categorization:
    return await llm.categorize(
        normalized=normalized,
        allowed=ALLOWED_CATEGORY_VALUES,
        user_prompt=None,
    )
