"""
Production mock LLM adapter. Vendor-keyed category; captures prompt in notes.
"""
import hashlib
from domain.models import AllowedCategory, Categorization, NormalizedReceipt
from application.ports import LLMPort


_VENDOR_CATEGORY = {
    "uber": AllowedCategory.TRAVEL,
    "delta": AllowedCategory.TRAVEL,
    "starbucks": AllowedCategory.MEALS,
    "staples": AllowedCategory.OFFICE_SUPPLIES,
    "fedex": AllowedCategory.SHIPPING,
    "aws": AllowedCategory.SOFTWARE,
    "coned": AllowedCategory.UTILITIES,
}


class MockLLMAdapter(LLMPort):
    async def categorize(self, normalized: NormalizedReceipt, allowed, user_prompt):
        vendor_key = (normalized.vendor or "").lower()
        category = _VENDOR_CATEGORY.get(vendor_key, AllowedCategory.OTHER)
        notes = f"mock categorization; prompt_seen={user_prompt!r}"
        seed = hashlib.sha256((vendor_key + (user_prompt or "")).encode()).hexdigest()
        confidence = 0.70 + (int(seed, 16) % 25) / 100
        return Categorization(
            category=category, confidence=round(confidence, 2),
            notes=notes, issues=[],
        )
