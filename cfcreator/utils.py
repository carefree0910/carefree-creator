import cv2
import math
import torch

import numpy as np

from enum import Enum
from typing import Any
from typing import List
from typing import Optional
from typing import Protocol
from cflearn.api.utils import ILoadableItem
from cflearn.api.utils import ILoadablePool

from .parameters import init_to_cpu
from .parameters import auto_lazy_load
from .parameters import need_change_device
from .parameters import weights_pool_limit


def resize_image(input_image: np.ndarray, resolution: int) -> np.ndarray:
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(
        input_image,
        (W, H),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA,
    )
    return img


def to_canvas(results: List[np.ndarray], *, padding: int = 0) -> np.ndarray:
    num_results = len(results)
    if num_results == 1:
        return results[0]
    num_col = math.ceil(math.sqrt(num_results))
    num_row = round(num_results / num_col)
    if num_row * num_col < num_results:
        num_row += 1
    h, w = results[0].shape[:2]
    canvas_w = num_col * w + (num_col - 1) * padding
    canvas_h = num_row * h + (num_row - 1) * padding
    canvas = np.full([canvas_h, canvas_w, 3], 255, np.uint8)
    for i, out in enumerate(results):
        ih, iw = out.shape[:2]
        if h != ih:
            raise ValueError(f"`h` mismatch: {ih} != {h}")
        if w != iw:
            raise ValueError(f"`w` mismatchh: {iw} != {w}")
        ix = i % num_col
        iy = i // num_col
        ix = ix * w + ix * padding
        iy = iy * h + iy * padding
        canvas[iy : iy + h, ix : ix + w] = out
    return canvas


class APIs(str, Enum):
    SD = "sd"
    SD_INPAINTING = "sd_inpainting"
    ESR = "esr"
    ESR_ANIME = "esr_anime"
    INPAINTING = "inpainting"
    LAMA = "lama"
    SEMANTIC = "semantic"
    HRNET = "hrnet"
    ISNET = "isnet"
    BLIP = "blip"
    PROMPT_ENHANCE = "prompt_enhance"


class IAPI:
    def to(self, device: str, *, use_half: bool) -> None:
        pass


class APIInit(Protocol):
    def __call__(self, init_to_cpu: bool) -> IAPI:
        pass


class LoadableAPI(ILoadableItem[IAPI]):
    force_not_lazy: bool = False

    def __init__(self, init_fn: APIInit, *, init: bool = False):
        super().__init__(lambda: init_fn(init_to_cpu() or self.lazy), init=init)

    @property
    def lazy(self) -> bool:
        return auto_lazy_load() and not self.force_not_lazy

    @property
    def need_change_device(self) -> bool:
        return need_change_device() or self.lazy

    def load(self, **kwargs: Any) -> IAPI:
        super().load()
        if not kwargs.get("no_change", False) and self.need_change_device:
            self._item.to("cuda:0", use_half=True, **kwargs)
        return self._item

    def cleanup(self, **kwargs: Any) -> None:
        if self.need_change_device:
            self._item.to("cpu", use_half=False, **kwargs)
            torch.cuda.empty_cache()

    def unload(self) -> None:
        self.cleanup()
        return super().unload()


class APIPool(ILoadablePool[IAPI]):
    def __init__(self) -> None:
        super().__init__(weights_pool_limit())

    def register(self, key: str, init_fn: APIInit) -> None:
        def _init(init: bool) -> LoadableAPI:
            api = LoadableAPI(init_fn, init=False)
            print("> init", key, "(lazy)" if api.lazy else "")
            if init:
                api.load()
            return api

        if key in self:
            return
        return super().register(key, _init)

    def cleanup(self, key: str, **kwargs: Any) -> None:
        loadable_api: Optional[LoadableAPI] = self.pool.get(key)
        if loadable_api is None:
            raise ValueError(f"key '{key}' does not exist")
        loadable_api.cleanup(**kwargs)

    def need_change_device(self, key: str) -> bool:
        loadable_api: Optional[LoadableAPI] = self.pool.get(key)
        if loadable_api is None:
            raise ValueError(f"key '{key}' does not exist")
        return loadable_api.need_change_device


api_pool = APIPool()
