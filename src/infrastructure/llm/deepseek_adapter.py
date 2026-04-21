"""
DeepSeek categorization adapter.

DeepSeek's chat API is OpenAI-compatible. We use AsyncOpenAI with a custom
base_url. The user prompt is injected verbatim into the system message.
Response format: JSON object matching Categorization.
"""
import asyncio
import json
from openai import AsyncOpenAI
from domain.models import AllowedCategory, Categorization, Issue, NormalizedReceipt
from application.ports import LLMPort


_SYSTEM_TEMPLATE = """\
You are an expense-category classifier. Given normalized receipt fields, choose
exactly ONE category from: {allowed}. If none fit, choose "Other" and provide a
note. Output strict JSON:

{{
  "category": "<one of the allowed categories>",
  "confidence": 0.0-1.0,
  "notes": "short rationale",
  "issues": [
    {{"severity": "warning", "code": "<short>", "message": "<explanation>"}}
  ]
}}

Issues to flag as warnings when relevant:
- missing receipt_number
- ambiguous currency
- total_out_of_range
- low_confidence

{user_instructions}
"""


class DeepSeekLLMAdapter(LLMPort):
    def __init__(self, *, api_key: str, base_url: str, model: str, timeout_s: int) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._timeout_s = timeout_s

    async def categorize(self, normalized: NormalizedReceipt, allowed, user_prompt) -> Categorization:
        system = _SYSTEM_TEMPLATE.format(
            allowed=", ".join(allowed),
            user_instructions=(f"Additional guidance from user: {user_prompt}" if user_prompt else ""),
        )
        user = json.dumps({
            "vendor": normalized.vendor,
            "receipt_date": normalized.receipt_date.isoformat() if normalized.receipt_date else None,
            "receipt_number": normalized.receipt_number,
            "total": str(normalized.total) if normalized.total else None,
            "currency": normalized.currency,
        })
        resp = await asyncio.wait_for(
            self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            ),
            timeout=self._timeout_s,
        )
        body = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise ValueError(f"categorizer returned non-JSON: {body[:200]!r}") from e
        issues = [Issue(**i) for i in data.get("issues", [])]
        return Categorization(
            category=AllowedCategory(data["category"]),
            confidence=float(data.get("confidence", 0.0)),
            notes=data.get("notes"),
            issues=issues,
        )
