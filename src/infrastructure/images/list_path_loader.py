"""Explicit-paths image loader.

Validates a list of caller-supplied paths against the assets_dir
guardrail and allowed extensions. All validation happens in __init__
so bad input is rejected before the graph runs.
"""
from __future__ import annotations
import os
from pathlib import Path

from application.ports import ImageLoaderPort, ImageRef


class TooManyPathsError(ValueError):
    """Distinct subclass so the HTTP layer can map it to 413."""


class ListPathImageLoader(ImageLoaderPort):
    def __init__(
        self,
        paths: list[str],
        allowed_extensions: set[str],
        assets_dir: Path,
    ) -> None:
        if len(paths) == 0:
            raise ValueError("image_paths must not be empty")

        allowed = {e.lower().lstrip(".") for e in allowed_extensions}
        assets_resolved = assets_dir.resolve()

        bad: list[tuple[str, str]] = []
        refs: list[ImageRef] = []
        for original in paths:
            p = Path(original).resolve()
            if not p.is_relative_to(assets_resolved):
                bad.append((original, f"path must be under {assets_resolved}"))
                continue
            if not p.exists():
                bad.append((original, "path does not exist"))
                continue
            if not p.is_file():
                bad.append((original, "path is not a file"))
                continue
            ext = p.suffix.lower().lstrip(".")
            if ext not in allowed:
                bad.append((original, f"extension .{ext} not in allowed extensions"))
                continue
            if not os.access(p, os.R_OK):
                bad.append((original, "path is not readable"))
                continue
            refs.append(ImageRef(source_ref=original, local_path=p))

        if bad:
            lines = [f"  {orig}: {why}" for orig, why in bad]
            raise ValueError("invalid image_paths:\n" + "\n".join(lines))

        self._refs = refs

    async def load(self) -> list[ImageRef]:
        return list(self._refs)
