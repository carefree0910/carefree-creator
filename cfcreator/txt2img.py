import time

from typing import Any
from fastapi import Response
from cfclient.models import AlgorithmBase
from cfcv.misc.toolkit import np_to_bytes

from .common import get_sd
from .common import Txt2ImgModel


txt2img_sd_endpoint = "/txt2img/sd"


@AlgorithmBase.register("txt2img.sd")
class Txt2ImgSD(AlgorithmBase):
    endpoint = txt2img_sd_endpoint

    def initialize(self) -> None:
        self.m = get_sd()

    async def run(self, data: Txt2ImgModel, *args: Any) -> Response:
        self.log_endpoint(data)
        t = time.time()
        size = data.w, data.h
        img_arr = self.m.txt2img(data.text, size=size, max_wh=data.max_wh).numpy()[0]
        img_arr = 0.5 * (img_arr + 1.0)
        img_arr = img_arr.transpose([1, 2, 0])
        self.log_times({"total": time.time() - t})
        return Response(content=np_to_bytes(img_arr), media_type="image/png")


__all__ = [
    "txt2img_sd_endpoint",
    "Txt2ImgSD",
]
