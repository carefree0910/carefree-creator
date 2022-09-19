import time

import numpy as np

from PIL import Image
from typing import Any
from typing import Dict
from typing import List
from fastapi import Response
from pydantic import Field
from cfclient.utils import download_image_with_retry
from cfclient.models import AlgorithmBase

from .common import get_esr
from .common import get_semantic
from .common import get_esr_anime
from .common import get_inpainting
from .common import get_bytes_from_diffusion
from .common import get_bytes_from_translator
from .common import Img2ImgModel


img2img_sr_endpoint = "/img2img/sr"
img2img_inpainting_endpoint = "/img2img/inpainting"
img2img_semantic2img_endpoint = "/img2img/semantic2img"


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


class Img2ImgSemantic2ImgModel(Img2ImgModel):
    color2label: Dict[str, int] = Field(
        ...,
        description="""
Mapping of color -> (semantic) label.
> The color should be of `rgb(r,g,b)` format.
""",
    )


def color2rgb(color: str) -> List[int]:
    if not color.startswith("rgb(") or not color.endswith(")"):
        raise ValueError("`color` should be of `rgb(r,g,b)` format")
    return [int(n.strip()) for n in color[4:-1].split(",")]


@AlgorithmBase.register("img2img.semantic2img")
class Img2ImgSemantic2Img(AlgorithmBase):
    endpoint = img2img_semantic2img_endpoint

    def initialize(self) -> None:
        self.m = get_semantic()

    async def run(self, data: Img2ImgSemantic2ImgModel, *args: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        raw_semantic = await download_image_with_retry(self.http_client.session, data.url)
        t1 = time.time()
        w, h = raw_semantic.size
        # `0` is the `unlabeled` class
        semantic_arr = np.zeros([h, w], np.uint8).ravel()
        raw_arr = np.array(raw_semantic)
        alpha = None
        valid_mask = None
        # handle alpha
        if raw_arr.shape[-1] == 4:
            alpha = raw_arr[..., -1]
            raw_arr = raw_arr[..., :3]
            valid_mask = alpha > 0
        for color, label in data.color2label.items():
            rgb = color2rgb(color)
            label_mask = (raw_arr == rgb).all(axis=2)
            if valid_mask is not None:
                label_mask &= valid_mask
            semantic_arr[label_mask.ravel()] = label
        semantic_arr = semantic_arr.reshape([h, w, 1]).repeat(3, axis=2)
        if alpha is not None:
            semantic_arr = np.concatenate([semantic_arr, alpha[..., None]], axis=2)
        semantic = Image.fromarray(semantic_arr)
        t2 = time.time()
        img_arr = self.m.semantic2img(semantic, max_wh=data.max_wh).numpy()[0]
        content = get_bytes_from_diffusion(img_arr)
        self.log_times({"download": t1 - t0, "preprocess": t2 - t1,  "inference": time.time() - t2})
        return Response(content=content, media_type="image/png")


__all__ = [
    "img2img_sr_endpoint",
    "img2img_inpainting_endpoint",
    "img2img_semantic2img_endpoint",
    "Img2ImgSRModel",
    "Img2ImgInpaintingModel",
    "Img2ImgSemantic2ImgModel",
    "Img2ImgSR",
    "Img2ImgInpainting",
    "Img2ImgSemantic2Img",
]
