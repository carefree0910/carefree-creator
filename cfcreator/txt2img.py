import time

from typing import Any
from fastapi import Response
from cfclient.models import AlgorithmBase

from .common import get_sd
from .common import handle_diffusion_model
from .common import get_bytes_from_diffusion
from .common import Txt2ImgModel


txt2img_sd_endpoint = "/txt2img/sd"


class Txt2ImgSDModel(Txt2ImgModel):
    w: int = Field(512, description="The desired output width.")
    h: int = Field(512, description="The desired output height.")


@AlgorithmBase.register("txt2img.sd")
class Txt2ImgSD(AlgorithmBase):
    endpoint = txt2img_sd_endpoint

    def initialize(self) -> None:
        self.m = get_sd()

    async def run(self, data: Txt2ImgSDModel, *args: Any) -> Response:
        self.log_endpoint(data)
        t = time.time()
        size = data.w, data.h
        kwargs = handle_diffusion_model(self.m, data)
        img_arr = self.m.txt2img(
            data.text,
            size=size,
            max_wh=data.max_wh,
            **kwargs,
        ).numpy()[0]
        content = get_bytes_from_diffusion(img_arr)
        self.log_times({"inference": time.time() - t})
        return Response(content=content, media_type="image/png")


__all__ = [
    "txt2img_sd_endpoint",
    "Txt2ImgSDModel",
    "Txt2ImgSD",
]
