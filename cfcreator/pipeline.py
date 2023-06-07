import time

import numpy as np

from PIL import Image
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from fastapi import Response
from pydantic import Field
from pydantic import BaseModel
from cftool.cv import to_rgb
from cftool.cv import to_uint8
from cftool.cv import np_to_bytes
from cfclient.models.core import ImageModel

from .cv import affine
from .cv import BaseAffineModel
from .utils import to_canvas
from .common import IAlgorithm
from .common import ReturnArraysModel


paste_pipeline_endpoint = "/pipeline/paste"


def get_response(data: ReturnArraysModel, results: List[np.ndarray]) -> Any:
    if data.return_arrays:
        return results
    return Response(content=np_to_bytes(to_canvas(results)), media_type="image/png")


# paste pipeline


def paste(
    original_fg: Image.Image,
    original_bg: Image.Image,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    force_rgb: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    t0 = time.time()
    if original_fg.mode != "RGBA":
        original_fg = original_fg.convert("RGBA")
    original_w, original_h = original_bg.size
    affined_fg_array = affine(
        np.array(original_fg),
        a,
        b,
        c,
        d,
        e,
        f,
        original_w,
        original_h,
    )
    t1 = time.time()
    affined_fg_array = affined_fg_array.astype(np.float32) / 255.0
    rgb = affined_fg_array[..., :3]
    mask = affined_fg_array[..., -1:]
    if force_rgb:
        original_bg = to_rgb(original_bg)
    bg_array = np.array(original_bg).astype(np.float32) / 255.0
    fg_array = rgb if bg_array.shape[2] == 3 else affined_fg_array
    merged = fg_array * mask + bg_array * (1.0 - mask)
    results = dict(rgb=rgb, mask=mask, merged=to_uint8(merged))
    latencies = {"affine": t1 - t0, "merge": time.time() - t1}
    return results, latencies


class _PastePipelineModel(BaseModel):
    bg_url: str = Field(
        ...,
        description="""
The `cdn` / `cos` url of the background's image.
> `cos` url from `qcloud` is preferred.
""",
    )
    force_rgb: bool = Field(False, description="Whether to force the output to be RGB.")
    return_mask: bool = Field(False, description="Whether to return the mask.")


class PastePipelineModel(
    ReturnArraysModel,
    BaseAffineModel,
    _PastePipelineModel,
    ImageModel,
):
    pass


@IAlgorithm.auto_register()
class PastePipeline(IAlgorithm):
    model_class = PastePipelineModel

    endpoint = paste_pipeline_endpoint

    def initialize(self) -> None:
        pass

    async def run(
        self,
        data: PastePipelineModel,
        *args: Any,
        **kwargs: Any,
    ) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        original_fg = await self.get_image_from("url", data, kwargs)
        original_bg = await self.get_image_from("bg_url", data, kwargs)
        t1 = time.time()
        results, latencies = paste(
            original_fg,
            original_bg,
            data.a,
            data.b,
            data.c,
            data.d,
            data.e,
            data.f,
            data.force_rgb,
        )
        latencies["download"] = t1 - t0
        self.log_times(latencies)
        if not data.return_mask:
            return get_response(data, [results["merged"]])
        mask = to_uint8(results["mask"])[..., 0]
        return get_response(data, [results["merged"], mask])


__all__ = [
    "paste_pipeline_endpoint",
    "PastePipelineModel",
    "PastePipeline",
]
