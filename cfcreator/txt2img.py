import time

from enum import Enum
from typing import Any
from fastapi import Response
from pydantic import Field
from cfclient.utils import download_image_with_retry
from cfclient.models import ImageModel
from cfclient.models import AlgorithmBase

from .common import get_sd
from .common import get_sd_anime
from .common import handle_diffusion_model
from .common import get_bytes_from_diffusion
from .common import IAlgorithm
from .common import Txt2ImgModel


txt2img_sd_endpoint = "/txt2img/sd"
txt2img_sd_outpainting_endpoint = "/txt2img/sd.outpainting"


class Txt2ImgSDModel(Txt2ImgModel):
    w: int = Field(512, description="The desired output width.")
    h: int = Field(512, description="The desired output height.")
    is_anime: bool = Field(False, description="Whether should we generate anime images or not.")


@AlgorithmBase.register("txt2img.sd")
class Txt2ImgSD(IAlgorithm):
    model_class = Txt2ImgSDModel

    endpoint = txt2img_sd_endpoint

    def initialize(self) -> None:
        self.m = get_sd()
        self.m_anime = get_sd_anime()

    async def run(self, data: Txt2ImgSDModel, *args: Any) -> Response:
        self.log_endpoint(data)
        t = time.time()
        size = data.w, data.h
        m = self.m_anime if data.is_anime else self.m
        kwargs = handle_diffusion_model(m, data)
        img_arr = m.txt2img(
            data.text,
            size=size,
            max_wh=data.max_wh,
            **kwargs,
        ).numpy()[0]
        content = get_bytes_from_diffusion(img_arr)
        self.log_times({"inference": time.time() - t})
        return Response(content=content, media_type="image/png")


class PaddingModes(str, Enum):
    CV2_NS = "cv2_ns"
    CV2_TELEA = "cv2_telea"


class Txt2ImgSDOutpaintingModel(Txt2ImgModel, ImageModel):
    fidelity: float = Field(0.2, description="The fidelity of the input image.")
    padding_mode: PaddingModes = Field(
        "cv2_telea",
        description="The outpainting padding mode.",
    )


@AlgorithmBase.register("txt2img.sd.outpainting")
class Txt2ImgSDOutpainting(IAlgorithm):
    model_class = Txt2ImgSDOutpaintingModel

    endpoint = txt2img_sd_outpainting_endpoint

    def initialize(self) -> None:
        self.m = get_sd()

    async def run(self, data: Txt2ImgSDOutpaintingModel, *args: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await download_image_with_retry(self.http_client.session, data.url)
        t1 = time.time()
        kwargs = handle_diffusion_model(self.m, data)
        img_arr = self.m.outpainting(
            data.text,
            image,
            anchor=64,
            max_wh=data.max_wh,
            fidelity=data.fidelity,
            padding_mode=data.padding_mode,
            **kwargs,
        ).numpy()[0]
        content = get_bytes_from_diffusion(img_arr)
        self.log_times({"download": t1 - t0, "inference": time.time() - t1})
        return Response(content=content, media_type="image/png")


__all__ = [
    "txt2img_sd_endpoint",
    "txt2img_sd_outpainting_endpoint",
    "Txt2ImgSDModel",
    "Txt2ImgSDOutpaintingModel",
    "Txt2ImgSD",
    "Txt2ImgSDOutpainting",
]
