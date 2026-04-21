from pathlib import Path
from application.ports import ImageLoaderPort, ImageRef


class MockImageLoader(ImageLoaderPort):
    def __init__(self, refs: list[ImageRef]) -> None:
        self._refs = refs

    async def load(self) -> list[ImageRef]:
        return list(self._refs)
