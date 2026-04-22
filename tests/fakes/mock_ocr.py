"""Deterministic OCR mock. Returns canned RawReceipt keyed by source_ref."""
from pathlib import Path
from domain.models import RawReceipt
from application.ports import OCRPort, ImageRef


class MockOCR(OCRPort):
    def __init__(self, responses: dict[str, RawReceipt] | None = None,
                 fail_on: set[str] | None = None) -> None:
        self._responses = responses or {}
        self._fail_on = fail_on or set()

    async def extract(self, image: ImageRef, hint: str | None = None) -> RawReceipt:
        if image.source_ref in self._fail_on:
            raise RuntimeError(f"mock OCR configured to fail on {image.source_ref}")
        if image.source_ref in self._responses:
            return self._responses[image.source_ref]
        # sensible default
        return RawReceipt(
            source_ref=image.source_ref,
            vendor="MockVendor",
            receipt_date="2024-03-15",
            receipt_number="R-MOCK-001",
            total_raw="$12.34",
            ocr_confidence=0.9,
        )
