from pathlib import Path
from application.ports import ImageLoaderPort, ImageRef


class LocalFolderImageLoader(ImageLoaderPort):
    def __init__(self, folder: Path, allowed_extensions: set[str]) -> None:
        self._folder = folder
        self._allowed = {e.lower().lstrip(".") for e in allowed_extensions}

    async def load(self) -> list[ImageRef]:
        if not self._folder.exists() or not self._folder.is_dir():
            return []
        refs: list[ImageRef] = []
        for p in sorted(self._folder.iterdir()):
            if p.is_file() and p.suffix.lower().lstrip(".") in self._allowed:
                refs.append(ImageRef(source_ref=p.name, local_path=p.resolve()))
        return refs
