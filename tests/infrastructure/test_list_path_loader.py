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


def test_rejects_adjacent_prefix_path(tmp_path):
    """A path that shares the assets_dir's string prefix but isn't under it is rejected."""
    assets = _make_assets(tmp_path)
    # e.g. assets=/tmp/.../assets, adjacent=/tmp/.../assets_evil/r1.png
    adjacent = tmp_path / "assets_evil"
    adjacent.mkdir()
    evil = adjacent / "r1.png"
    evil.write_bytes(b"\x89PNG")
    with pytest.raises(ValueError, match="must be under"):
        ListPathImageLoader([str(evil)], ALLOWED, assets)


def test_rejects_empty_list(tmp_path):
    with pytest.raises(ValueError, match="must not be empty"):
        ListPathImageLoader([], ALLOWED, tmp_path)


def test_rejects_path_outside_assets(tmp_path):
    assets = _make_assets(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"\x89PNG")
    with pytest.raises(ValueError, match="must be under"):
        ListPathImageLoader([str(outside)], ALLOWED, assets)


def test_rejects_missing_file(tmp_path):
    assets = _make_assets(tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        ListPathImageLoader([str(assets / "ghost.png")], ALLOWED, assets)


def test_rejects_directory(tmp_path):
    assets = _make_assets(tmp_path)
    subdir = assets / "sub"
    subdir.mkdir()
    with pytest.raises(ValueError, match="is not a file"):
        ListPathImageLoader([str(subdir)], ALLOWED, assets)


def test_rejects_disallowed_extension(tmp_path):
    assets = _make_assets(tmp_path)
    exe = assets / "script.exe"
    exe.write_bytes(b"MZ")
    with pytest.raises(ValueError, match=r"extension \.exe"):
        ListPathImageLoader([str(exe)], ALLOWED, assets)


def test_aggregates_all_failures(tmp_path):
    assets = _make_assets(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"\x89PNG")
    exe = assets / "script.exe"
    exe.write_bytes(b"MZ")
    with pytest.raises(ValueError) as exc_info:
        ListPathImageLoader(
            [str(outside), str(assets / "ghost.png"), str(exe)],
            ALLOWED, assets,
        )
    msg = str(exc_info.value)
    # All three bad paths reported in one message
    assert "outside.png" in msg
    assert "ghost.png" in msg
    assert "script.exe" in msg
    assert msg.count("\n") >= 3  # header line + 3 reason lines
