import time

import numpy as np

from PIL import Image
from enum import Enum
from typing import Any
from typing import Optional
from fastapi import Response
from pydantic import Field
from pydantic import BaseModel
from cftool.cv import to_uint8
from cfclient.models import ImageModel

from .utils import api_pool
from .utils import APIs
from .common import register_sd
from .common import register_sd_inpainting
from .common import get_sd_from
from .common import handle_diffusion_model
from .common import get_bytes_from_diffusion
from .common import get_normalized_arr_from_diffusion
from .common import handle_diffusion_inpainting_model
from .common import IAlgorithm
from .common import HighresModel
from .common import Txt2ImgModel
from .common import ReturnArraysModel
from .common import CommonSDInpaintingModel


txt2img_sd_endpoint = "/txt2img/sd"
txt2img_sd_inpainting_endpoint = "/txt2img/sd.inpainting"
txt2img_sd_outpainting_endpoint = "/txt2img/sd.outpainting"


class _Txt2ImgSDModel(BaseModel):
    w: int = Field(512, description="The desired output width.")
    h: int = Field(512, description="The desired output height.")
    highres_info: Optional[HighresModel] = Field(None, description="Highres info.")


class Txt2ImgSDModel(ReturnArraysModel, Txt2ImgModel, _Txt2ImgSDModel):
    pass


@IAlgorithm.auto_register()
class Txt2ImgSD(IAlgorithm):
    model_class = Txt2ImgSDModel

    endpoint = txt2img_sd_endpoint

    def initialize(self) -> None:
        register_sd()

    async def run(self, data: Txt2ImgSDModel, *args: Any, **kwargs: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        m = get_sd_from(APIs.SD, data)
        t1 = time.time()
        size = data.w, data.h
        kwargs.update(handle_diffusion_model(m, data))
        if data.highres_info is not None:
            kwargs["highres_info"] = data.highres_info.dict()
        img_arr = m.txt2img(
            data.text,
            size=size,
            max_wh=data.max_wh,
            **kwargs,
        ).numpy()[0]
        if data.return_arrays:
            content = None
        else:
            content = get_bytes_from_diffusion(img_arr)
        t2 = time.time()
        api_pool.cleanup(APIs.SD)
        self.log_times(
            {
                "get_model": t1 - t0,
                "inference": t2 - t1,
                "cleanup": time.time() - t2,
            }
        )
        if content is None:
            return [to_uint8(get_normalized_arr_from_diffusion(img_arr))]
        return Response(content=content, media_type="image/png")


class PaddingModes(str, Enum):
    CV2_NS = "cv2_ns"
    CV2_TELEA = "cv2_telea"


class Txt2ImgSDInpaintingModel(CommonSDInpaintingModel, Txt2ImgModel, ImageModel):
    mask_url: str = Field(
        ...,
        description="""
The `cdn` / `cos` url of the user's mask.
> `cos` url from `qcloud` is preferred.
> If empty string is provided, then we will use an empty mask, which means we will simply perform an image-to-image transform.  
""",
    )


class Txt2ImgSDOutpaintingModel(CommonSDInpaintingModel, Txt2ImgModel, ImageModel):
    pass


@IAlgorithm.auto_register()
class Txt2ImgSDInpainting(IAlgorithm):
    model_class = Txt2ImgSDInpaintingModel

    endpoint = txt2img_sd_inpainting_endpoint

    def initialize(self) -> None:
        register_sd()
        register_sd_inpainting()

    async def run(
        self,
        data: Txt2ImgSDInpaintingModel,
        *args: Any,
        **kwargs: Any,
    ) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        mask = await self.get_image_from("mask_url", data, kwargs)
        t1 = time.time()
        if data.use_raw_inpainting:
            api_key = APIs.SD
        else:
            api_key = APIs.SD_INPAINTING
        m = get_sd_from(api_key, data)
        t2 = time.time()
        kwargs.update(handle_diffusion_model(m, data))
        kwargs.update(handle_diffusion_inpainting_model(data))
        mask_arr = np.array(mask)
        mask_arr[..., -1] = np.where(mask_arr[..., -1] > 0, 255, 0)
        mask = Image.fromarray(mask_arr)
        img_arr = m.txt2img_inpainting(
            data.text,
            image,
            mask,
            anchor=64,
            max_wh=data.max_wh,
            keep_original=data.keep_original,
            **kwargs,
        ).numpy()[0]
        if data.return_arrays:
            content = None
        else:
            content = get_bytes_from_diffusion(img_arr)
        t3 = time.time()
        api_pool.cleanup(api_key)
        self.log_times(
            {
                "download": t1 - t0,
                "get_model": t2 - t1,
                "inference": t3 - t2,
                "cleanup": time.time() - t3,
            }
        )
        if content is None:
            return [to_uint8(get_normalized_arr_from_diffusion(img_arr))]
        return Response(content=content, media_type="image/png")


@IAlgorithm.auto_register()
class Txt2ImgSDOutpainting(IAlgorithm):
    model_class = Txt2ImgSDOutpaintingModel

    endpoint = txt2img_sd_outpainting_endpoint

    def initialize(self) -> None:
        register_sd_inpainting()

    async def run(
        self,
        data: Txt2ImgSDOutpaintingModel,
        *args: Any,
        **kwargs: Any,
    ) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        m = get_sd_from(APIs.SD_INPAINTING, data)
        m.disable_control()
        t2 = time.time()
        kwargs.update(handle_diffusion_model(m, data))
        kwargs.update(handle_diffusion_inpainting_model(data))
        img_arr = m.outpainting(
            data.text,
            image,
            anchor=64,
            max_wh=data.max_wh,
            keep_original=data.keep_original,
            **kwargs,
        ).numpy()[0]
        if data.return_arrays:
            content = None
        else:
            content = get_bytes_from_diffusion(img_arr)
        t3 = time.time()
        api_pool.cleanup(APIs.SD_INPAINTING)
        self.log_times(
            {
                "download": t1 - t0,
                "get_model": t2 - t1,
                "inference": t3 - t2,
                "cleanup": time.time() - t3,
            }
        )
        if content is None:
            return [to_uint8(get_normalized_arr_from_diffusion(img_arr))]
        return Response(content=content, media_type="image/png")


__all__ = [
    "txt2img_sd_endpoint",
    "txt2img_sd_inpainting_endpoint",
    "txt2img_sd_outpainting_endpoint",
    "Txt2ImgSDModel",
    "Txt2ImgSDInpaintingModel",
    "Txt2ImgSDOutpaintingModel",
    "Txt2ImgSD",
    "Txt2ImgSDInpainting",
    "Txt2ImgSDOutpainting",
]
