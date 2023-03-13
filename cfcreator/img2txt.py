import time

from PIL import Image
from typing import Any
from cfcv.misc.toolkit import to_rgb
from cflearn.api.cv.models.common import restrict_wh
from cflearn.api.cv.third_party.blip import BLIPAPI

from .common import IAlgorithm
from .common import TextModel
from .common import ImageModel
from .common import MaxWHModel
from .parameters import init_to_cpu
from .parameters import need_change_device


img2txt_caption_endpoint = "/img2txt/caption"


class Img2TxtModel(MaxWHModel, ImageModel):
    pass


@IAlgorithm.auto_register()
class Img2TxtCaption(IAlgorithm):
    model_class = Img2TxtModel
    response_model_class = TextModel

    endpoint = img2txt_caption_endpoint

    def initialize(self) -> None:
        print("> init blip")
        self.m = BLIPAPI("cpu" if init_to_cpu() else "cuda:0")

    async def run(self, data: Img2TxtModel, *args: Any) -> TextModel:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.download_image_with_retry(data.url)
        t1 = time.time()
        w, h = image.size
        w, h = restrict_wh(w, h, data.max_wh)
        image = image.resize((w, h), resample=Image.LANCZOS)
        t2 = time.time()
        if need_change_device():
            self.m.to("cuda:0")
        t3 = time.time()
        caption = self.m.caption(to_rgb(image))
        t4 = time.time()
        if need_change_device():
            self.m.to("cpu")
        self.log_times(
            {
                "download": t1 - t0,
                "preprocess": t2 - t1,
                "get_model": t3 - t2,
                "inference": t4 - t3,
                "cleanup": time.time() - t4,
            }
        )
        return TextModel(text=caption)


__all__ = [
    "img2txt_caption_endpoint",
    "Img2TxtCaption",
]
