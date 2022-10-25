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
from cfcv.misc.toolkit import to_rgb

from .common import cleanup
from .common import get_esr
from .common import init_sd_ms
from .common import get_sd_from
from .common import get_semantic
from .common import get_esr_anime
from .common import get_inpainting
from .common import handle_diffusion_model
from .common import get_bytes_from_diffusion
from .common import get_bytes_from_translator
from .common import IAlgorithm
from .common import Img2ImgModel
from .common import Img2ImgDiffusionModel
from .parameters import save_gpu_ram


img2img_sd_endpoint = "/img2img/sd"
img2img_sr_endpoint = "/img2img/sr"
img2img_inpainting_endpoint = "/img2img/inpainting"
img2img_semantic2img_endpoint = "/img2img/semantic2img"


class Img2ImgSDModel(Img2ImgDiffusionModel):
    text: str = Field(..., description="The text that we want to handle.")
    fidelity: float = Field(0.2, description="The fidelity of the input image.")
    keep_alpha: bool = Field(True, description="""
Whether the returned image should keep the alpha-channel of the input image or not.
> If the input image is a sketch image, then `keep_alpha` needs to be False in most of the time.  
"""
    )
    is_anime: bool = Field(False, description="Whether should we generate anime images or not.")


@IAlgorithm.auto_register()
class Img2ImgSD(IAlgorithm):
    model_class = Img2ImgSDModel

    endpoint = img2img_sd_endpoint

    def initialize(self) -> None:
        self.ms = init_sd_ms()

    async def run(self, data: Img2ImgSDModel, *args: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await download_image_with_retry(self.http_client.session, data.url)
        t1 = time.time()
        if not data.keep_alpha:
            image = to_rgb(image)
        m = get_sd_from(self.ms, data)
        t2 = time.time()
        kwargs = handle_diffusion_model(m, data)
        img_arr = m.img2img(
            image,
            cond=[data.text],
            max_wh=data.max_wh,
            fidelity=data.fidelity,
            anchor=64,
            **kwargs,
        ).numpy()[0]
        content = get_bytes_from_diffusion(img_arr)
        t3 = time.time()
        cleanup(m)
        self.log_times(
            {
                "download": t1 - t0,
                "get_model": t2 - t1,
                "inference": t3 - t2,
                "cleanup": time.time() - t3,
            }
        )
        return Response(content=content, media_type="image/png")


class Img2ImgSRModel(Img2ImgModel):
    is_anime: bool = Field(False, description="Whether the input image is an anime image or not.")


@IAlgorithm.auto_register()
class Img2ImgSR(IAlgorithm):
    model_class = Img2ImgSRModel

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


class Img2ImgInpaintingModel(Img2ImgDiffusionModel):
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
> If empty string is provided, then we will use an empty mask, which means we will simply perform an image-to-image transform.  
""",
    )


@IAlgorithm.auto_register()
class Img2ImgInpainting(IAlgorithm):
    model_class = Img2ImgInpaintingModel

    endpoint = img2img_inpainting_endpoint

    def initialize(self) -> None:
        self.m = get_inpainting()

    async def run(self, data: Img2ImgInpaintingModel, *args: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await download_image_with_retry(self.http_client.session, data.url)
        mask_url = data.mask_url
        if not mask_url:
            mask = Image.new("L", image.size, color=0)
        else:
            mask = await download_image_with_retry(self.http_client.session, mask_url)
        t1 = time.time()
        if save_gpu_ram():
            self.m.to("cuda:0", use_half=True)
        t2 = time.time()
        refine_fidelity = data.refine_fidelity if data.use_refine else None
        img_arr = self.m.inpainting(
            image,
            mask,
            max_wh=data.max_wh,
            refine_fidelity=refine_fidelity,
        ).numpy()[0]
        content = get_bytes_from_diffusion(img_arr)
        t3 = time.time()
        cleanup(self.m)
        self.log_times(
            {
                "download": t1 - t0,
                "get_model": t2 - t1,
                "inference": t3 - t2,
                "cleanup": time.time() - t3,
            }
        )
        return Response(content=content, media_type="image/png")


class Img2ImgSemantic2ImgModel(Img2ImgDiffusionModel):
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


@IAlgorithm.auto_register()
class Img2ImgSemantic2Img(IAlgorithm):
    model_class = Img2ImgSemantic2ImgModel

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
        if save_gpu_ram():
            self.m.to("cuda:0", use_half=True)
        t4 = time.time()
        if not data.keep_alpha:
            alpha = None
        elif alpha is not None:
            alpha = alpha[None, None].astype(np.float32) / 255.0
        img_arr = self.m.semantic2img(semantic, alpha=alpha, max_wh=data.max_wh).numpy()[0]
        content = get_bytes_from_diffusion(img_arr)
        t5 = time.time()
        cleanup(self.m)
        self.log_times(
            {
                "download": t1 - t0,
                "preprocess": t2 - t1,
                "interpolation": t3 - t2,
                "get_model": t4 - t3,
                "inference": t5 - t4,
                "cleanup": time.time() - t5,
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
