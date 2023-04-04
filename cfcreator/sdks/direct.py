# `direct` sdk is used to access to the server launched by `cfcreator serve`

from cfcreator import *
from typing import Any
from typing import Dict
from PIL.Image import Image

from .utils import *


async def txt2img(model: Txt2ImgSDModel) -> Image:
    return await get_image_res(get_url(txt2img_sd_endpoint), model)


async def push_direct(endpoint: str, params: Dict[str, Any]) -> Image:
    return await get_image_res(get_url(endpoint), params)


__all__ = [
    "txt2img",
]
