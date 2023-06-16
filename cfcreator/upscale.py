import time

import numpy as np

from PIL import Image
from PIL import ImageDraw
from typing import Any
from typing import List
from typing import Tuple
from typing import Optional
from fastapi import Response
from pydantic import Field
from cftool.misc import random_hash
from cftool.misc import shallow_copy_dict
from cflearn.api.cv.diffusion import TPair

from .common import get_response
from .common import Txt2ImgModel
from .common import InpaintingMode
from .common import IWrapperAlgorithm
from .common import ReturnArraysModel
from .control_multi import TBundle
from .control_multi import ControlMultiModel


upscale_tile_endpoint = "/upscale/tile"


class UpscaleTileModel(ReturnArraysModel, Txt2ImgModel):
    url: str = Field(..., description="url of the initial image")
    padding: int = Field(32, description="padding for each tile")
    grid_wh: TPair = Field(None, description="explicit specify the grid size")
    upscale_factor: TPair = Field(2, ge=1, description="upscale factor")
    fidelity: float = Field(0.45, description="fidelity of each tile")
    highres_steps: int = Field(36, description="num_steps for upscaling")
    strength: float = Field(1.0, description="strength of the tile control")
    controls: Optional[List[TBundle]] = Field(None, description="Extra controls")


def resize(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return image.resize(size, Image.LANCZOS)


@IWrapperAlgorithm.auto_register()
class UpscaleTile(IWrapperAlgorithm):
    model_class = UpscaleTileModel

    endpoint = upscale_tile_endpoint

    async def run(self, data: UpscaleTileModel, *args: Any, **kwargs: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        canvas = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        if data.grid_wh is None:
            w_grid, h_grid = canvas.size
        else:
            grid_wh = data.grid_wh
            if isinstance(grid_wh, int):
                grid_wh = grid_wh, grid_wh
            w_grid, h_grid = grid_wh
        factor = data.upscale_factor
        if isinstance(factor, int):
            factor = factor, factor
        w_factor, h_factor = factor
        w, h = w_grid * w_factor, h_grid * h_factor
        canvas = resize(canvas, (w, h))
        for k, v in kwargs.items():
            if k != "url" and isinstance(v, Image.Image):
                kwargs[k] = resize(v, (w, h))
        all_black = Image.new("RGB", (w, h), color=(0, 0, 0))
        all_black_draw = ImageDraw.Draw(all_black)
        controlnet_data = ControlMultiModel(
            url=random_hash(),
            mask_url=random_hash(),
            prompt=data.text,
            base_model=data.version,
            seed=data.seed,
            sampler=data.sampler,
            inpainting_mode=InpaintingMode.MASKED,
            inpainting_target_wh=(w_grid, h_grid),
            controls=[
                dict(
                    type="control_v11f1e_sd15_tile",
                    data=dict(
                        hint_url=random_hash(),
                        control_strength=data.strength,
                    ),
                )
            ],
            keep_original=False,
            # params
            num_steps=data.highres_steps,
            use_reference=True,
            reference_fidelity=data.fidelity,
            inpainting_mask_padding=data.padding,
        )
        if data.controls is not None:
            controlnet_data.controls = data.controls + controlnet_data.controls
        n_controls = len(controlnet_data.controls)
        t2 = time.time()
        for j in range(h_factor):
            jy = j % h_factor * h_grid
            for i in range(w_factor):
                ix = i % w_factor * w_grid
                lt_rb = ix, jy, ix + w_grid, jy + h_grid
                all_black_draw.rectangle(lt_rb, fill=(255, 255, 255))
                kw = shallow_copy_dict(kwargs)
                kw.update(
                    {
                        "url": canvas,
                        "mask_url": all_black,
                        f"controls.{n_controls - 1}.data.hint_url": canvas,
                    }
                )
                images = await self.apis.run_multi_controlnet(controlnet_data, **kw)
                canvas = images[-1]
                all_black_draw.rectangle(lt_rb, fill=(0, 0, 0))
        t3 = time.time()
        res = get_response(data, [np.array(canvas)])
        latencies = {
            "download": t1 - t0,
            "preprocess": t2 - t1,
            "inference": t3 - t2,
            "get_response": time.time() - t3,
        }
        self.log_times(latencies)
        return res


__all__ = [
    "upscale_tile_endpoint",
    "UpscaleTileModel",
    "UpscaleTile",
]
