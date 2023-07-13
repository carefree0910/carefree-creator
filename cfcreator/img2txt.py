import time

from PIL import Image
from typing import Any
from cftool.cv import to_rgb
from cftool.cv import restrict_wh

from .utils import api_pool
from .utils import APIs
from .common import register_blip
from .common import IAlgorithm
from .common import TextModel
from .common import ImageModel
from .common import MaxWHModel


img2txt_caption_endpoint = "/img2txt/caption"


class Img2TxtModel(MaxWHModel, ImageModel):
    pass


@IAlgorithm.auto_register()
class Img2TxtCaption(IAlgorithm):
    model_class = Img2TxtModel
    response_model_class = TextModel

    endpoint = img2txt_caption_endpoint

    def initialize(self) -> None:
        register_blip()

    async def run(self, data: Img2TxtModel, *args: Any, **kwargs: Any) -> TextModel:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        w, h = image.size
        w, h = restrict_wh(w, h, data.max_wh)
        image = image.resize((w, h), resample=Image.LANCZOS)
        t2 = time.time()
        m = api_pool.get(APIs.BLIP)
        t3 = time.time()
        caption = m.caption(to_rgb(image))
        self.log_times(
            {
                "download": t1 - t0,
                "preprocess": t2 - t1,
                "get_model": t3 - t2,
                "inference": time.time() - t3,
            }
        )
        return TextModel(text=caption)


__all__ = [
    "img2txt_caption_endpoint",
    "Img2TxtModel",
    "Img2TxtCaption",
]
