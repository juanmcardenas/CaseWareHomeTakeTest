"""
OpenAI vision OCR adapter.

Sends the image as a base64 data URL to a vision-capable model (default: gpt-4o-mini)
and asks for strict JSON matching our RawReceipt schema.
"""
import asyncio
import base64
import json
from pathlib import Path
from openai import AsyncOpenAI
from domain.models import RawReceipt
from application.ports import OCRPort, ImageRef


_SYSTEM = (
    "You are an OCR extractor for expense receipts. Read the image and return a "
    "single JSON object with these fields (all optional strings unless noted): "
    "vendor (string), receipt_date (string as it appears), receipt_number (string), "
    "total_raw (string, include currency symbol if present), currency_raw (string, "
    "ISO 4217 if you can tell), line_items (array of objects with description/amount), "
    "ocr_confidence (number 0.0-1.0 - your self-assessed confidence). "
    "Return ONLY valid JSON with no prose."
)


class OpenAIOCRAdapter(OCRPort):
    def __init__(self, *, api_key: str, model: str, timeout_s: int) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._timeout_s = timeout_s

    async def extract(self, image: ImageRef, hint: str | None = None) -> RawReceipt:
        system_text = _SYSTEM
        if hint:
            system_text = f"{system_text}\n\nAdditional hint: {hint}"
        data_url = _image_to_data_url(image.local_path)
        resp = await asyncio.wait_for(
            self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract the receipt."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]},
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
            raise ValueError(f"OCR returned non-JSON: {body[:200]!r}") from e
        return RawReceipt(source_ref=image.source_ref, **_safe_subset(data))


def _image_to_data_url(p: Path) -> str:
    raw = p.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    ext = p.suffix.lstrip(".").lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "png")
    return f"data:image/{mime};base64,{b64}"


_ALLOWED_KEYS = {
    "vendor", "receipt_date", "receipt_number", "total_raw",
    "currency_raw", "line_items", "ocr_confidence",
}


def _safe_subset(data: dict) -> dict:
    return {k: v for k, v in data.items() if k in _ALLOWED_KEYS}
