import time

from typing import Any
from cfcv.misc.toolkit import to_rgb
from cflearn.api.cv.third_party.blip import BLIPAPI

from .common import IAlgorithm
from .common import TextModel
from .common import ImageModel
from .parameters import init_to_cpu
from .parameters import need_change_device


img2txt_caption_endpoint = "/img2txt/caption"


@IAlgorithm.auto_register()
class Img2TxtCaption(IAlgorithm):
    model_class = ImageModel
    response_model_class = TextModel

    endpoint = img2txt_caption_endpoint

    def initialize(self) -> None:
        print("> init blip")
        self.m = BLIPAPI("cpu" if init_to_cpu() else "cuda:0")

    async def run(self, data: ImageModel, *args: Any) -> TextModel:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.download_image_with_retry(data.url)
        t1 = time.time()
        if need_change_device():
            self.m.to("cuda:0")
        t2 = time.time()
        caption = self.m.caption(to_rgb(image))
        t3 = time.time()
        if need_change_device():
            self.m.to("cpu")
        self.log_times(
            {
                "download": t1 - t0,
                "get_model": t2 - t1,
                "inference": t3 - t2,
                "cleanup": time.time() - t3,
            }
        )
        return TextModel(text=caption)


__all__ = [
    "img2txt_caption_endpoint",
    "Img2TxtCaption",
]
