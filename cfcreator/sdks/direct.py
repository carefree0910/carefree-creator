# `direct` sdk is used to access to the server launched by `cfcreator serve`

from cfcreator import *
from typing import Any
from typing import Dict
from PIL.Image import Image

from .utils import *


class DirectSDK:
    def __init__(self, host: str = "http://localhost:8123") -> None:
        self.host = host

    async def txt2img(self, model: Txt2ImgSDModel) -> Image:
        return await self.push_direct(txt2img_sd_endpoint, model.dict())

    async def push_direct(self, endpoint: str, params: Dict[str, Any]) -> Image:
        return await get_image_res(get_url(self.host, endpoint), params)


__all__ = [
    "DirectSDK",
]
