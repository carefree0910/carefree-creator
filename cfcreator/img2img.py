import cv2
import time

import numpy as np

from PIL import Image
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from fastapi import Response
from pydantic import Field
from pydantic import BaseModel
from scipy.interpolate import NearestNDInterpolator
from cftool.cv import to_rgb
from cftool.cv import to_uint8
from cftool.cv import read_image
from cftool.cv import np_to_bytes
from cflearn.api.cv import TranslatorAPI
from cflearn.api.cv.third_party.lama import Config

from .utils import api_pool
from .utils import APIs
from .common import register_sd
from .common import register_esr
from .common import register_lama
from .common import register_hrnet
from .common import register_isnet
from .common import register_semantic
from .common import register_esr_anime
from .common import register_inpainting
from .common import get_sd_from
from .common import get_response
from .common import handle_diffusion_model
from .common import get_normalized_arr_from_diffusion
from .common import IAlgorithm
from .common import ImageModel
from .common import HighresModel
from .common import Img2ImgModel
from .common import CallbackModel
from .common import ReturnArraysModel
from .common import Img2ImgDiffusionModel
from .parameters import verbose
from .parameters import get_focus
from .parameters import Focus


img2img_sd_endpoint = "/img2img/sd"
img2img_sr_endpoint = "/img2img/sr"
img2img_inpainting_endpoint = "/img2img/inpainting"
img2img_semantic2img_endpoint = "/img2img/semantic2img"
img2img_harmonization_endpoint = "/img2img/harmonization"
img2img_sod_endpoint = "/img2img/sod"


# img2img (stable diffusion)


class _Img2ImgSDModel(BaseModel):
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
    highres_info: Optional[HighresModel] = Field(None, description="Highres info.")


class Img2ImgSDModel(ReturnArraysModel, Img2ImgDiffusionModel, _Img2ImgSDModel):
    pass


@IAlgorithm.auto_register()
class Img2ImgSD(IAlgorithm):
    model_class = Img2ImgSDModel

    endpoint = img2img_sd_endpoint

    def initialize(self) -> None:
        register_sd()

    async def run(self, data: Img2ImgSDModel, *args: Any, **kwargs: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        if not data.keep_alpha:
            image = to_rgb(image)
        w, h = data.wh
        if w > 0 and h > 0:
            image = image.resize((w, h), Image.LANCZOS)
        t2 = time.time()
        m = get_sd_from(APIs.SD, data)
        t3 = time.time()
        kwargs.update(handle_diffusion_model(m, data))
        if data.highres_info is not None:
            kwargs["highres_info"] = data.highres_info.dict()
        img_arr = m.img2img(
            image,
            cond=[data.text],
            max_wh=data.max_wh,
            fidelity=data.fidelity,
            anchor=64,
            **kwargs,
        ).numpy()[0]
        t4 = time.time()
        res = get_response(data, [to_uint8(get_normalized_arr_from_diffusion(img_arr))])
        self.log_times(
            {
                "download": t1 - t0,
                "preprocess": t2 - t1,
                "get_model": t3 - t2,
                "inference": t4 - t3,
                "get_response": time.time() - t4,
            }
        )
        return res


# super resolution (Real-ESRGAN)


class SRVersion(str, Enum):
    ULTRASHARP = "ultrasharp"


class _Img2ImgSRModel(BaseModel):
    is_anime: bool = Field(
        False,
        description="Whether the input image is an anime image or not.",
    )
    version: Optional[SRVersion] = Field(None, description="The explicit version.")
    target_w: int = Field(0, description="The target width. 0 means as-is.")
    target_h: int = Field(0, description="The target height. 0 means as-is.")


class Img2ImgSRModel(ReturnArraysModel, CallbackModel, _Img2ImgSRModel, Img2ImgModel):
    max_wh: int = Field(832, description="The maximum resolution.")


def apply_sr(
    m: TranslatorAPI,
    image: Image.Image,
    max_wh: int,
    target_w: int,
    target_h: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    t0 = time.time()
    img_arr = m.sr(image, max_wh=max_wh).numpy()[0]
    img_arr = img_arr.transpose([1, 2, 0])
    t1 = time.time()
    h, w = img_arr.shape[:2]
    if target_w and target_h:
        larger = w * h < target_w * target_h
        img_arr = cv2.resize(
            img_arr,
            (target_w, target_h),
            interpolation=cv2.INTER_LANCZOS4 if larger else cv2.INTER_AREA,
        )
    elif target_w or target_h:
        if target_w:
            k = target_w / w
            target_h = round(h * k)
        else:
            k = target_h / h
            target_w = round(w * k)
        img_arr = cv2.resize(
            img_arr,
            (target_w, target_h),
            interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA,
        )
    return img_arr, dict(inference=t1 - t0, resize=time.time() - t1)


@IAlgorithm.auto_register()
class Img2ImgSR(IAlgorithm):
    model_class = Img2ImgSRModel

    endpoint = img2img_sr_endpoint

    def initialize(self) -> None:
        register_esr()
        register_esr_anime()

    async def run(self, data: Img2ImgSRModel, *args: Any, **kwargs: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        if data.version is None:
            api_key = APIs.ESR_ANIME if data.is_anime else APIs.ESR
        else:
            if data.version == SRVersion.ULTRASHARP:
                api_key = APIs.ESR_ULTRASHARP
            else:
                raise ValueError(f"Unknown version: {data.version}")
        m = api_pool.get(api_key)
        t2 = time.time()
        img_arr, latencies = apply_sr(
            m,
            image,
            data.max_wh,
            data.target_w,
            data.target_h,
        )
        t3 = time.time()
        res = get_response(data, [to_uint8(img_arr)])
        latencies.update(
            {
                "download": t1 - t0,
                "get_model": t2 - t1,
                "get_response": time.time() - t3,
            }
        )
        self.log_times(latencies)
        return res


# inpainting (LDM, LaMa)


class InpaintingModels(str, Enum):
    SD = "sd"
    LAMA = "lama"


class Img2ImgInpaintingModel(ReturnArraysModel, Img2ImgDiffusionModel):
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
    max_wh: int = Field(832, description="The maximum resolution.")


@IAlgorithm.auto_register()
class Img2ImgInpainting(IAlgorithm):
    model_class = Img2ImgInpaintingModel

    endpoint = img2img_inpainting_endpoint

    def initialize(self) -> None:
        focus = get_focus()
        self.is_sync = focus == Focus.SYNC
        if not self.is_sync:
            register_inpainting()
        register_lama()

    async def run(
        self,
        data: Img2ImgInpaintingModel,
        *args: Any,
        **kwargs: Any,
    ) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        mask_url = data.mask_url
        if not mask_url and "mask_url" not in kwargs:
            mask = Image.new("L", image.size, color=0)
        else:
            mask = await self.get_image_from("mask_url", data, kwargs)
        t1 = time.time()
        model = data.model
        api_key = APIs.LAMA if model == InpaintingModels.LAMA else APIs.INPAINTING
        m = api_pool.get(api_key)
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
            img_arr = m(image_arr, mask_arr, cfg)
            final = to_uint8(img_arr)
        else:
            kwargs.update(handle_diffusion_model(m, data))
            mask_arr = np.array(mask)
            mask_arr[..., -1] = np.where(mask_arr[..., -1] > 0, 255, 0)
            mask = Image.fromarray(mask_arr)
            if not data.use_pipeline:
                refine_fidelity = data.refine_fidelity if data.use_refine else None
                img_arr = m.inpainting(
                    image,
                    mask,
                    max_wh=data.max_wh,
                    refine_fidelity=refine_fidelity,
                    **kwargs,
                ).numpy()[0]
            else:
                img_arr = m.inpainting(
                    image,
                    mask,
                    max_wh=data.max_wh,
                    refine_fidelity=None,
                    **kwargs,
                ).numpy()[0]
                img_arr = get_normalized_arr_from_diffusion(img_arr)
                image = Image.fromarray(to_uint8(img_arr))
                img_arr = m.inpainting(
                    image,
                    mask,
                    max_wh=data.max_wh,
                    refine_fidelity=data.refine_fidelity,
                    **kwargs,
                ).numpy()[0]
            final = to_uint8(get_normalized_arr_from_diffusion(img_arr))
        res = get_response(data, [final])
        self.log_times(
            {
                "download": t1 - t0,
                "get_model": t2 - t1,
                "inference": time.time() - t2,
            }
        )
        return res


# semantic2img (LDM)


class Img2ImgSemantic2ImgModel(ReturnArraysModel, Img2ImgDiffusionModel):
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
        register_semantic()

    async def run(
        self,
        data: Img2ImgSemantic2ImgModel,
        *args: Any,
        **kwargs: Any,
    ) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        raw_semantic = await self.get_image_from("url", data, kwargs)
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
        m = api_pool.get(APIs.SEMANTIC)
        t4 = time.time()
        if not data.keep_alpha:
            alpha = None
        elif alpha is not None:
            alpha = alpha[None, None].astype(np.float32) / 255.0
        img_arr = m.semantic2img(
            semantic,
            alpha=alpha,
            max_wh=data.max_wh,
            verbose=verbose(),
            seed=data.seed,
            **kwargs,
        ).numpy()[0]
        t5 = time.time()
        res = get_response(data, [to_uint8(get_normalized_arr_from_diffusion(img_arr))])
        self.log_times(
            {
                "download": t1 - t0,
                "preprocess": t2 - t1,
                "interpolation": t3 - t2,
                "get_model": t4 - t3,
                "inference": t5 - t4,
                "get_response": time.time() - t5,
            }
        )
        return res


# image harmonization (hrnet)


def apply_harmonization(
    max_wh: int,
    strength: float,
    raw_image: np.ndarray,
    normalized_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    t0 = time.time()
    m = api_pool.get(APIs.HRNET)
    t1 = time.time()
    h, w = raw_image.shape[:2]
    scale = max_wh**2 / (w * h)
    if scale >= 1.0:
        scaled_image = raw_image
        scaled_mask = normalized_mask
    else:
        scaled_w = round(w * scale)
        scaled_h = round(h * scale)
        scaled_image = cv2.resize(raw_image, (scaled_w, scaled_h))
        scaled_mask = cv2.resize(normalized_mask, (scaled_w, scaled_h))
    result = m.predict(scaled_image, scaled_mask)
    if scale < 1.0:
        result = cv2.resize(result, (w, h))
    if strength != 1.0:
        raw_image = raw_image.astype(np.float32)
        result = result.astype(np.float32)
        result = result * strength + raw_image * (1.0 - strength)
        result = (np.clip(result, 0.0, 255.0)).astype(np.uint8)
    latencies = {
        "get_model": t1 - t0,
        "inference": time.time() - t1,
    }
    return result, latencies


class Img2ImgHarmonizationModel(ReturnArraysModel, ImageModel):
    mask_url: str = Field(
        ...,
        description="The `cdn` / `cos` url of the harmonization mask. (`cos` url is preferred)",
    )
    strength: float = Field(1.0, description="Strength of the harmonization process.")
    harmonization_max_wh: int = Field(
        2048,
        description="max_wh for the harmonization inputs.",
    )


@IAlgorithm.auto_register()
class Img2ImgHarmonization(IAlgorithm):
    model_class = Img2ImgHarmonizationModel

    endpoint = img2img_harmonization_endpoint

    def initialize(self) -> None:
        register_hrnet()

    async def run(
        self,
        data: Img2ImgHarmonizationModel,
        *args: Any,
        **kwargs: Any,
    ) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        mask = await self.get_image_from("mask_url", data, kwargs)
        t1 = time.time()
        mask_arr = read_image(
            mask,
            None,
            anchor=None,
            to_mask=True,
            to_torch_fmt=False,
        ).image
        mask_arr[mask_arr > 0.0] = 1.0
        result, latencies = apply_harmonization(
            data.harmonization_max_wh,
            data.strength,
            read_image(
                image,
                None,
                anchor=None,
                normalize=False,
                to_torch_fmt=False,
            ).image,
            mask_arr,
        )
        latencies["download"] = t1 - t0
        self.log_times(latencies)
        if data.return_arrays:
            return [result]
        return Response(content=np_to_bytes(result), media_type="image/png")


# salient object detection (isnet)


class Img2ImgSODModel(ReturnArraysModel, ImageModel):
    pass


@IAlgorithm.auto_register()
class Img2ImgSOD(IAlgorithm):
    model_class = Img2ImgSODModel

    endpoint = img2img_sod_endpoint

    def initialize(self) -> None:
        register_isnet()

    async def run(self, data: Img2ImgSODModel, *args: Any, **kwargs: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        m = api_pool.get(APIs.ISNET)
        t2 = time.time()
        rgb = to_rgb(image)
        alpha = to_uint8(m.segment(rgb))
        content = None if data.return_arrays else np_to_bytes(alpha)
        self.log_times(
            {
                "download": t1 - t0,
                "get_model": t2 - t1,
                "inference": time.time() - t2,
            }
        )
        if content is None:
            return [np.concatenate([np.array(rgb), alpha[..., None]], axis=2)]
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
    "Img2ImgSODModel",
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
