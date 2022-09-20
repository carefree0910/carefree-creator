import time

import numpy as np

from PIL import Image
from typing import Any
from typing import Dict
from typing import List
from fastapi import Response
from pydantic import Field
from scipy.interpolate import NearestNDInterpolator
from cfclient.utils import download_image_with_retry
from cfclient.models import AlgorithmBase

from .common import get_sd
from .common import get_esr
from .common import get_semantic
from .common import get_esr_anime
from .common import get_inpainting
from .common import get_bytes_from_diffusion
from .common import get_bytes_from_translator
from .common import Img2ImgModel


img2img_sd_endpoint = "/img2img/sd"
img2img_sr_endpoint = "/img2img/sr"
img2img_inpainting_endpoint = "/img2img/inpainting"
img2img_semantic2img_endpoint = "/img2img/semantic2img"


class Img2ImgSDModel(Img2ImgModel):
    text: str = Field(..., description="The text that we want to handle.")
    fidelity: float = Field(0.2, description="The fidelity of the input image.")


@AlgorithmBase.register("img2img.sd")
class Img2ImgSD(AlgorithmBase):
    endpoint = img2img_sd_endpoint

    def initialize(self) -> None:
        self.m = get_sd()

    async def run(self, data: Img2ImgSDModel, *args: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await download_image_with_retry(self.http_client.session, data.url)
        t1 = time.time()
        img_arr = self.m.img2img(
            image,
            cond=[data.text],
            max_wh=data.max_wh,
            fidelity=data.fidelity,
            anchor=64,
        ).numpy()[0]
        content = get_bytes_from_diffusion(img_arr)
        self.log_times({"download": t1 - t0, "inference": time.time() - t1})
        return Response(content=content, media_type="image/png")


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
    keep_alpha: bool = Field(
        False,
        description="Whether the returned image should keep the "
                    "alpha-channel of the input image or not.",
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
        raw_arr = np.array(raw_semantic)
        alpha = None
        valid_mask = None
        valid_mask_ravel = None
        # handle alpha
        if raw_arr.shape[-1] == 4:
            alpha = raw_arr[..., -1]
            raw_arr = raw_arr[..., :3]
            valid_mask = alpha > 0
            valid_mask_ravel = valid_mask.ravel()
        # get nearest color
        raw_arr_flat = raw_arr.reshape([h * w, 1, 3])
        if valid_mask_ravel is not None:
            raw_arr_flat = raw_arr_flat[valid_mask_ravel]
        raw_arr_flat = raw_arr_flat.astype(np.int16)
        colors = sorted(data.color2label)
        rgbs = np.array(list(map(color2rgb, colors)), np.int16).reshape([1, -1, 3])
        diff = np.abs(raw_arr_flat - rgbs).mean(2)
        indices = np.argmin(diff, axis=1)
        # diffusion has no `unlabeled` label, so it should be COCO.label - 1
        labels = np.array([data.color2label[color] - 1 for color in colors], np.uint8)
        arr_labels = labels[indices]
        if valid_mask_ravel is None:
            semantic_arr = arr_labels
        else:
            semantic_arr = np.zeros([h, w], np.uint8).ravel()
            semantic_arr[valid_mask_ravel] = arr_labels
        # nearest interpolation
        t2 = time.time()
        if valid_mask is not None and valid_mask_ravel is not None:
            to_coordinates = lambda mask: np.array(np.nonzero(mask)).T
            valid_coordinates = to_coordinates(valid_mask)
            interpolator = NearestNDInterpolator(valid_coordinates, arr_labels)
            invalid_mask = ~valid_mask
            invalid_coordinates = to_coordinates(invalid_mask)
            semantic_arr[invalid_mask.ravel()] = interpolator(invalid_coordinates)
        # gather
        semantic_arr = semantic_arr.reshape([h, w])
        semantic = Image.fromarray(semantic_arr)
        t3 = time.time()
        if not data.keep_alpha:
            alpha = None
        elif alpha is not None:
            alpha = alpha[None, None].astype(np.float32) / 255.0
        img_arr = self.m.semantic2img(semantic, alpha=alpha, max_wh=data.max_wh).numpy()[0]
        content = get_bytes_from_diffusion(img_arr)
        self.log_times(
            {
                "download": t1 - t0,
                "preprocess": t2 - t1,
                "interpolation": t3 - t2,
                "inference": time.time() - t3,
            }
        )
        return Response(content=content, media_type="image/png")


__all__ = [
    "img2img_sd_endpoint",
    "img2img_sr_endpoint",
    "img2img_inpainting_endpoint",
    "img2img_semantic2img_endpoint",
    "Img2ImgSDModel",
    "Img2ImgSRModel",
    "Img2ImgInpaintingModel",
    "Img2ImgSemantic2ImgModel",
    "Img2ImgSD",
    "Img2ImgSR",
    "Img2ImgInpainting",
    "Img2ImgSemantic2Img",
]
