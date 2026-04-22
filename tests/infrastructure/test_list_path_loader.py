"""Unit tests for ListPathImageLoader — validates a list of paths against
the ASSETS_DIR guardrail and allowed extensions."""
from pathlib import Path
import os
import pytest

from infrastructure.images.list_path_loader import (
    ListPathImageLoader, TooManyPathsError,
)


ALLOWED = {"jpg", "jpeg", "png", "webp", "pdf"}


def _make_assets(tmp_path: Path) -> Path:
    """Create a tmp assets dir with valid images and return the dir path."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "r1.png").write_bytes(b"\x89PNG\r\n\x1a\nfakepng")
    (assets / "r2.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpg")
    return assets


@pytest.mark.asyncio
async def test_loads_valid_paths_under_assets(tmp_path):
    assets = _make_assets(tmp_path)
    paths = [str(assets / "r1.png"), str(assets / "r2.jpg")]
    loader = ListPathImageLoader(paths, ALLOWED, assets)
    refs = await loader.load()
    assert len(refs) == 2
    assert refs[0].source_ref == paths[0]
    assert refs[1].source_ref == paths[1]
    assert refs[0].local_path == (assets / "r1.png").resolve()
    assert refs[1].local_path == (assets / "r2.jpg").resolve()
