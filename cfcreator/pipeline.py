import time

import numpy as np

from PIL import Image
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from fastapi import Response
from pydantic import Field
from cfcv.misc.toolkit import to_rgb
from cfcv.misc.toolkit import to_uint8
from cfcv.misc.toolkit import np_to_bytes
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
    a: int,
    b: int,
    c: int,
    d: int,
    e: int,
    f: int,
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
    rgb = affined_fg_array[..., :3].astype(np.float32) / 255.0
    mask = affined_fg_array[..., -1:].astype(np.float32) / 255.0
    original_bg_array = np.array(to_rgb(original_bg)).astype(np.float32) / 255.0
    merged = rgb * mask + original_bg_array * (1.0 - mask)
    results = dict(rgb=rgb, mask=mask, merged=to_uint8(merged))
    latencies = {"affine": t1 - t0, "merge": time.time() - t1}
    return results, latencies


class PastePipelineModel(ImageModel, BaseAffineModel, ReturnArraysModel):
    bg_url: str = Field(
        ...,
        description="""
The `cdn` / `cos` url of the background's image.
> `cos` url from `qcloud` is preferred.
""",
    )


@IAlgorithm.auto_register()
class PastePipeline(IAlgorithm):
    model_class = PastePipelineModel

    endpoint = paste_pipeline_endpoint

    def initialize(self) -> None:
        pass

    async def run(self, data: PastePipelineModel, *args: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        original_fg = await self.download_image_with_retry(data.url)
        original_bg = await self.download_image_with_retry(data.bg_url)
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
        )
        latencies["download"] = t1 - t0
        self.log_times(latencies)
        return get_response(data, [results["merged"]])


__all__ = [
    "paste_pipeline_endpoint",
    "PastePipelineModel",
    "PastePipeline",
]
