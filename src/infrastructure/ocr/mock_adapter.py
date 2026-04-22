"""
Production mock OCR adapter. Deterministic by filename.
"""
import hashlib
from domain.models import RawReceipt
from application.ports import OCRPort, ImageRef


_VENDORS = ["Uber", "Delta", "Starbucks", "Staples", "FedEx", "AWS", "ConEd"]


class MockOCRAdapter(OCRPort):
    async def extract(self, image: ImageRef, hint: str | None = None) -> RawReceipt:
        h = int(hashlib.sha256(image.source_ref.encode()).hexdigest(), 16)
        vendor = _VENDORS[h % len(_VENDORS)]
        total_cents = 500 + (h % 9500)
        total = f"${total_cents / 100:.2f}"
        return RawReceipt(
            source_ref=image.source_ref,
            vendor=vendor,
            receipt_date="2024-03-15",
            receipt_number=f"R-{h % 100000:05d}",
            total_raw=total,
            ocr_confidence=0.92,
        )
