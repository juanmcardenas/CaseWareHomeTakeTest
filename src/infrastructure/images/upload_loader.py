"""Save multipart uploads to a per-run temp directory; enforces max_size_bytes."""
import tempfile
from pathlib import Path
from fastapi import UploadFile
from application.ports import ImageLoaderPort, ImageRef


class UploadImageLoader(ImageLoaderPort):
    def __init__(self, uploads: list[UploadFile], allowed_extensions: set[str],
                 max_size_bytes: int) -> None:
        self._uploads = uploads
        self._allowed = {e.lower().lstrip(".") for e in allowed_extensions}
        self._max_size_bytes = max_size_bytes
        self._tmpdir = Path(tempfile.mkdtemp(prefix="receipt-run-"))

    async def load(self) -> list[ImageRef]:
        refs: list[ImageRef] = []
        for up in self._uploads:
            if not up.filename:
                continue
            ext = Path(up.filename).suffix.lower().lstrip(".")
            if ext not in self._allowed:
                continue
            content = await up.read()
            if len(content) > self._max_size_bytes:
                raise ValueError(
                    f"{up.filename}: exceeds max size {self._max_size_bytes} bytes"
                )
            dest = self._tmpdir / up.filename
            dest.write_bytes(content)
            refs.append(ImageRef(source_ref=up.filename, local_path=dest.resolve()))
        return refs
