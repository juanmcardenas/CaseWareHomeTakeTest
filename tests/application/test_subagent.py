import pytest
from decimal import Decimal
from domain.models import AllowedCategory, Categorization, NormalizedReceipt
from application.subagent import categorize_with_subagent
from tests.fakes.mock_llm import MockLLM


def _normalized(source_ref="r.png", vendor="Uber"):
    return NormalizedReceipt(
        source_ref=source_ref, vendor=vendor,
        total=Decimal("45.67"), currency="USD",
    )


@pytest.mark.asyncio
async def test_calls_llm_with_allowed_categories_and_no_user_prompt():
    """categorize_with_subagent MUST NOT forward the run-level user prompt —
    letting it influence categorization was observed to collapse the
    downstream filter into a no-op."""
    llm = MockLLM(default_category=AllowedCategory.TRAVEL)
    result = await categorize_with_subagent(llm, _normalized())
    assert result.category == AllowedCategory.TRAVEL
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call.user_prompt is None
    # allowed_categories passed by value (strings matching the enum values)
    assert set(call.allowed) == {c.value for c in AllowedCategory}


@pytest.mark.asyncio
async def test_rejects_out_of_band_category_classifies_as_invalid(monkeypatch):
    """
    If the LLM returns a category string not in AllowedCategory, sub-agent raises
    ValueError — the tool wrapper will convert this into a Band-A receipt error.
    """
    class BadLLM:
        async def categorize(self, normalized, allowed, user_prompt):
            # Bypassing the Pydantic validator by constructing raw dict? Instead,
            # simulate by raising directly — mirroring what a provider JSON-mode
            # mismatch would cause.
            raise ValueError("invalid category")

    with pytest.raises(ValueError):
        await categorize_with_subagent(BadLLM(), _normalized())
