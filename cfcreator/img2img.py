import time
import torch

import numpy as np

from PIL import Image
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from fastapi import Response
from pydantic import Field
from scipy.interpolate import NearestNDInterpolator
from cfcv.misc.toolkit import to_rgb
from cfcv.misc.toolkit import to_uint8
from cfcv.misc.toolkit import np_to_bytes
from cflearn.api.cv import ImageHarmonizationAPI
from cflearn.api.cv.models.common import read_image
from cflearn.api.cv.third_party.lama import LaMa
from cflearn.api.cv.third_party.lama import Config
from cflearn.api.cv.third_party.isnet import ISNetAPI

from .common import cleanup
from .common import get_esr
from .common import get_hrnet
from .common import init_sd
from .common import get_sd_from
from .common import get_semantic
from .common import get_esr_anime
from .common import get_inpainting
from .common import handle_diffusion_model
from .common import get_bytes_from_diffusion
from .common import get_bytes_from_translator
from .common import get_normalized_arr_from_diffusion
from .common import IAlgorithm
from .common import ImageModel
from .common import Img2ImgModel
from .common import CallbackModel
from .common import Img2ImgDiffusionModel
from .parameters import verbose
from .parameters import get_focus
from .parameters import init_to_cpu
from .parameters import auto_lazy_load
from .parameters import need_change_device
from .parameters import Focus


img2img_sd_endpoint = "/img2img/sd"
img2img_sr_endpoint = "/img2img/sr"
img2img_inpainting_endpoint = "/img2img/inpainting"
img2img_semantic2img_endpoint = "/img2img/semantic2img"
img2img_harmonization_endpoint = "/img2img/harmonization"
img2img_sod_endpoint = "/img2img/sod"


# img2img (stable diffusion)


class Img2ImgSDModel(Img2ImgDiffusionModel):
    text: str = Field(..., description="The text that we want to handle.")
    fidelity: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="The fidelity of the input image.",
    )
    keep_alpha: bool = Field(
        True,
        description="""
Whether the returned image should keep the alpha-channel of the input image or not.
> If the input image is a sketch image, then `keep_alpha` needs to be False in most of the time.  
""",
    )
    wh: Tuple[int, int] = Field(
        (0, 0),
        description="The output size, `0` means as-is",
    )


@IAlgorithm.auto_register()
class Img2ImgSD(IAlgorithm):
    model_class = Img2ImgSDModel

    endpoint = img2img_sd_endpoint

    def initialize(self) -> None:
        self.sd = init_sd()

    async def run(self, data: Img2ImgSDModel, *args: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.download_image_with_retry(data.url)
        t1 = time.time()
        if not data.keep_alpha:
            image = to_rgb(image)
        w, h = data.wh
        if w > 0 and h > 0:
            image = image.resize((w, h), Image.LANCZOS)
        t2 = time.time()
        m = get_sd_from(self.sd, data)
        t3 = time.time()
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
        t4 = time.time()
        cleanup(m)
        self.log_times(
            {
                "download": t1 - t0,
                "preprocess": t2 - t1,
                "get_model": t3 - t2,
                "inference": t4 - t3,
                "cleanup": time.time() - t4,
            }
        )
        return Response(content=content, media_type="image/png")


# super resolution (Real-ESRGAN)


class Img2ImgSRModel(Img2ImgModel, CallbackModel):
    is_anime: bool = Field(
        False,
        description="Whether the input image is an anime image or not.",
    )


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
        image = await self.download_image_with_retry(data.url)
        t1 = time.time()
        m = self.esr_anime if data.is_anime else self.esr
        if need_change_device():
            m.to("cuda:0", use_half=True)
        t2 = time.time()
        img_arr = m.sr(image, max_wh=data.max_wh).numpy()[0]
        content = get_bytes_from_translator(img_arr)
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


# inpainting (LDM, LaMa)


class InpaintingModels(str, Enum):
    SD = "sd"
    LAMA = "lama"


class Img2ImgInpaintingModel(Img2ImgDiffusionModel):
    model: InpaintingModels = Field(
        InpaintingModels.SD,
        description="The inpainting model that we want to use.",
    )
    use_refine: bool = Field(False, description="Whether should we perform refining.")
    use_pipeline: bool = Field(
        False,
        description="Whether should we perform 'inpainting' + 'refining' in one run.",
    )
    refine_fidelity: float = Field(
        0.2,
        description="""
Refine fidelity used in inpainting.
> Only take effects when `use_refine` / `use_pipeline` is set to True.
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
        focus = get_focus()
        self.lazy = auto_lazy_load()
        self.m = None if focus == Focus.SYNC else get_inpainting(self.lazy)
        print(f"> init lama{' (lazy)' if self.lazy else ''}")
        self.lama = LaMa("cpu" if init_to_cpu() or self.lazy else "cuda:0")

    async def run(self, data: Img2ImgInpaintingModel, *args: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.download_image_with_retry(data.url)
        model = data.model
        if self.m is None and model == InpaintingModels.SD:
            msg = "`sd` inpainting is not available when `focus` is set to 'sync'"
            raise ValueError(msg)
        mask_url = data.mask_url
        if not mask_url:
            mask = Image.new("L", image.size, color=0)
        else:
            mask = await self.download_image_with_retry(mask_url)
        t1 = time.time()
        if need_change_device() or self.lazy:
            if model == InpaintingModels.SD:
                self.m.to("cuda:0", use_half=True)
            elif model == InpaintingModels.LAMA:
                self.lama.to("cuda:0")
        t2 = time.time()
        if model == InpaintingModels.LAMA:
            cfg = Config()
            image_arr = read_image(
                image,
                None,
                anchor=None,
                to_torch_fmt=False,
            ).image
            mask_arr = read_image(
                mask,
                None,
                anchor=None,
                to_mask=True,
                to_torch_fmt=False,
            ).image
            mask_arr[mask_arr > 0.0] = 1.0
            result = self.lama(image_arr, mask_arr, cfg)
            content = np_to_bytes(result)
        else:
            kwargs = handle_diffusion_model(self.m, data)
            if not data.use_pipeline:
                refine_fidelity = data.refine_fidelity if data.use_refine else None
                img_arr = self.m.inpainting(
                    image,
                    mask,
                    max_wh=data.max_wh,
                    refine_fidelity=refine_fidelity,
                    **kwargs,
                ).numpy()[0]
            else:
                img_arr = self.m.inpainting(
                    image,
                    mask,
                    max_wh=data.max_wh,
                    refine_fidelity=None,
                    **kwargs,
                ).numpy()[0]
                img_arr = get_normalized_arr_from_diffusion(img_arr)
                image = Image.fromarray(to_uint8(img_arr))
                img_arr = self.m.inpainting(
                    image,
                    mask,
                    max_wh=data.max_wh,
                    refine_fidelity=data.refine_fidelity,
                    **kwargs,
                ).numpy()[0]
            content = get_bytes_from_diffusion(img_arr)
        t3 = time.time()
        if need_change_device() or self.lazy:
            self.lama.to("cpu")
            torch.cuda.empty_cache()
        if self.m is not None:
            cleanup(self.m, self.lazy)
        self.log_times(
            {
                "download": t1 - t0,
                "get_model": t2 - t1,
                "inference": t3 - t2,
                "cleanup": time.time() - t3,
            }
        )
        return Response(content=content, media_type="image/png")


# semantic2img (LDM)


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
        description="Whether the returned image should keep the alpha-channel of the input image or not.",
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
        self.lazy = auto_lazy_load()
        self.m = get_semantic(self.lazy)

    async def run(self, data: Img2ImgSemantic2ImgModel, *args: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        raw_semantic = await self.download_image_with_retry(data.url)
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
        if need_change_device() or self.lazy:
            self.m.to("cuda:0", use_half=True)
        t4 = time.time()
        if not data.keep_alpha:
            alpha = None
        elif alpha is not None:
            alpha = alpha[None, None].astype(np.float32) / 255.0
        img_arr = self.m.semantic2img(
            semantic,
            alpha=alpha,
            max_wh=data.max_wh,
            verbose=verbose(),
        ).numpy()[0]
        content = get_bytes_from_diffusion(img_arr)
        t5 = time.time()
        cleanup(self.m, self.lazy)
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


# image harmonization (hrnet)


def apply_harmonization(
    m: ImageHarmonizationAPI,
    strength: float,
    raw_image: np.ndarray,
    normalized_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.time()
    if need_change_device():
        m.to("cuda:0")
    t1 = time.time()
    result = m.run(raw_image, normalized_mask)
    if strength != 1.0:
        raw_image = raw_image.astype(np.float32)
        result = result.astype(np.float32)
        result = result * strength + raw_image * (1.0 - strength)
        result = (np.clip(result, 0.0, 255.0)).astype(np.uint8)
    t2 = time.time()
    if need_change_device():
        m.to("cpu")
        torch.cuda.empty_cache()
    latencies = {
        "get_model": t1 - t0,
        "inference": t2 - t1,
        "cleanup": time.time() - t2,
    }
    return result, latencies


class Img2ImgHarmonizationModel(ImageModel):
    mask_url: str = Field(
        ...,
        description="The `cdn` / `cos` url of the harmonization mask. (`cos` url is preferred)",
    )
    strength: float = Field(1.0, description="Strength of the harmonization process.")


@IAlgorithm.auto_register()
class Img2ImgHarmonization(IAlgorithm):
    model_class = Img2ImgHarmonizationModel

    endpoint = img2img_harmonization_endpoint

    def initialize(self) -> None:
        print("> init hrnet")
        self.m = get_hrnet()

    async def run(self, data: Img2ImgHarmonizationModel, *args: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.download_image_with_retry(data.url)
        mask = await self.download_image_with_retry(data.mask_url)
        t1 = time.time()
        result, latencies = apply_harmonization(
            self.m,
            data.strength,
            read_image(
                image,
                None,
                anchor=None,
                normalize=False,
                to_torch_fmt=False,
            ).image,
            read_image(
                mask,
                None,
                anchor=None,
                to_mask=True,
                to_torch_fmt=False,
            ).image,
        )
        latencies["download"] = t1 - t0
        self.log_times(latencies)
        return Response(content=np_to_bytes(result), media_type="image/png")


# salient object detection (isnet)


@IAlgorithm.auto_register()
class Img2ImgSOD(IAlgorithm):
    model_class = ImageModel

    endpoint = img2img_sod_endpoint

    def initialize(self) -> None:
        print("> init isnet")
        self.m = ISNetAPI("cpu" if init_to_cpu() else "cuda:0")

    async def run(self, data: ImageModel, *args: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.download_image_with_retry(data.url)
        t1 = time.time()
        if need_change_device():
            self.m.to("cuda:0")
        t2 = time.time()
        alpha = to_uint8(self.m.segment(image))
        content = np_to_bytes(alpha)
        t3 = time.time()
        if need_change_device():
            self.m.to("cpu")
            torch.cuda.empty_cache()
        self.log_times(
            {
                "download": t1 - t0,
                "get_model": t2 - t1,
                "inference": t3 - t2,
                "cleanup": time.time() - t3,
            }
        )
        return Response(content=content, media_type="image/png")


__all__ = [
    "img2img_sd_endpoint",
    "img2img_sr_endpoint",
    "img2img_inpainting_endpoint",
    "img2img_semantic2img_endpoint",
    "img2img_harmonization_endpoint",
    "img2img_sod_endpoint",
    "Img2ImgSDModel",
    "Img2ImgSRModel",
    "Img2ImgInpaintingModel",
    "Img2ImgSemantic2ImgModel",
    "Img2ImgSD",
    "Img2ImgSR",
    "Img2ImgInpainting",
    "Img2ImgSemantic2Img",
    "Img2ImgHarmonizationModel",
    "Img2ImgHarmonization",
    "Img2ImgSOD",
]
