import time

from typing import Any
from fastapi import Response
from pydantic import Field
from cfclient.utils import download_image_with_retry
from cfclient.models import AlgorithmBase

from .common import get_esr
from .common import get_esr_anime
from .common import get_inpainting
from .common import get_bytes_from_diffusion
from .common import get_bytes_from_translator
from .common import Img2ImgModel


img2img_sr_endpoint = "/img2img/sr"
img2img_inpainting_endpoint = "/img2img/inpainting"


class Img2ImgSRModel(Img2ImgModel):
    is_anime: bool = Field(False, description="Whether the input image is an anime image or not.")


@AlgorithmBase.register("img2img.sr")
class Img2ImgSR(AlgorithmBase):
    endpoint = img2img_sr_endpoint

    def initialize(self) -> None:
        self.esr = get_esr()
        self.esr_anime = get_esr_anime()

    async def run(self, data: Img2ImgSRModel, *args: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await download_image_with_retry(self.http_client.session, data.url)
        t1 = time.time()
        m = self.esr_anime if data.is_anime else self.esr
        img_arr = m.sr(image, max_wh=data.max_wh).numpy()[0]
        content = get_bytes_from_translator(img_arr)
        self.log_times({"download": t1 - t0, "inference": time.time() - t1})
        return Response(content=content, media_type="image/png")


class Img2ImgInpaintingModel(Img2ImgModel):
    use_refine: bool = Field(False, description="Whether should we perform refining.")
    refine_fidelity: float = Field(
        0.8,
        description="""
Refine fidelity used in inpainting.
> Only take effects when `use_refine` is set to True.
""",
    )
    mask_url: str = Field(
        ...,
        description="""
The `cdn` / `cos` url of the user's mask.
> `cos` url from `qcloud` is preferred.
""",
    )


@AlgorithmBase.register("img2img.inpainting")
class Img2ImgInpainting(AlgorithmBase):
    endpoint = img2img_inpainting_endpoint

    def initialize(self) -> None:
        self.m = get_inpainting()

    async def run(self, data: Img2ImgInpaintingModel, *args: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await download_image_with_retry(self.http_client.session, data.url)
        mask = await download_image_with_retry(self.http_client.session, data.mask_url)
        t1 = time.time()
        refine_fidelity = data.refine_fidelity if data.use_refine else None
        img_arr = self.m.inpainting(
            image,
            mask,
            max_wh=data.max_wh,
            refine_fidelity=refine_fidelity,
        ).numpy()[0]
        content = get_bytes_from_diffusion(img_arr)
        self.log_times({"download": t1 - t0, "inference": time.time() - t1})
        return Response(content=content, media_type="image/png")


__all__ = [
    "img2img_sr_endpoint",
    "img2img_inpainting_endpoint",
    "Img2ImgInpaintingModel",
    "Img2ImgSRModel",
    "Img2ImgInpainting",
    "Img2ImgSR",
]
