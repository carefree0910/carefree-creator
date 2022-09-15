import time

from typing import Any
from fastapi import Response
from cfclient.utils import download_image_with_retry
from cfclient.models import AlgorithmBase

from .common import get_sr
from .common import get_bytes_from_diffusion
from .common import Img2ImgModel


img2img_sr_endpoint = "/img2img/sr"


@AlgorithmBase.register("img2img.sr")
class Img2ImgSR(AlgorithmBase):
    endpoint = img2img_sr_endpoint

    def initialize(self) -> None:
        self.m = get_sr()

    async def run(self, data: Img2ImgModel, *args: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await download_image_with_retry(self.http_client.session, data.url)
        t1 = time.time()
        img_arr = self.m.sr(image, max_wh=data.max_wh).numpy()[0]
        content = get_bytes_from_diffusion(img_arr)
        self.log_times({"download": t1 - t0, "inference": time.time() - t1})
        return Response(content=content, media_type="image/png")


__all__ = [
    "img2img_sr_endpoint",
    "Img2ImgSR",
]
