import cv2
import math
import torch

import numpy as np

from PIL import Image
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Protocol
from cftool.cv import to_rgb
from cflearn.api.utils import ILoadableItem
from cflearn.api.utils import ILoadablePool

from .parameters import lazy_load
from .parameters import pool_limit
from .parameters import OPT


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


def get_contrast_bg(rgba_image: Image.Image) -> int:
    rgba = np.array(rgba_image)
    rgb = rgba[..., :3]
    alpha = rgba[..., -1]
    target_mask = alpha >= 10
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    lightness = hls[..., 1].astype(np.float32) / 255.0
    target_lightness = lightness[target_mask]
    mean = target_lightness.mean().item()
    std = target_lightness.std().item()
    if 0.45 <= mean <= 0.55 and std >= 0.25:
        return 127
    if mean <= 0.2 or 0.8 <= mean:
        return 127
    return 0 if mean >= 0.5 else 255


def to_contrast_rgb(image: Image.Image) -> Image.Image:
    if not image.mode == "RGBA":
        return to_rgb(image)
    bg = get_contrast_bg(image)
    return to_rgb(image, (bg, bg, bg))


class APIs(str, Enum):
    SD = "sd"
    SD_INPAINTING = "sd_inpainting"
    ESR = "esr"
    ESR_ANIME = "esr_anime"
    ESR_ULTRASHARP = "esr_ultrasharp"
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
    def __init__(
        self,
        init_fn: APIInit,
        *,
        init: bool = False,
        force_not_lazy: bool = False,
        has_annotator: bool = False,
    ):
        super().__init__(lambda: init_fn(self.init_to_cpu), init=init)
        self.force_not_lazy = force_not_lazy
        self.has_annotator = has_annotator

    @property
    def lazy(self) -> bool:
        return lazy_load() and not self.force_not_lazy

    @property
    def init_to_cpu(self) -> bool:
        return self.lazy or OPT["cpu"]

    @property
    def need_change_device(self) -> bool:
        return self.lazy and not OPT["cpu"]

    @property
    def annotator_kwargs(self) -> Dict[str, Any]:
        return {"no_annotator": True} if self.has_annotator else {}

    def load(self, *, no_change: bool = False, **kwargs: Any) -> IAPI:
        super().load()
        if not no_change and self.need_change_device and torch.cuda.is_available():
            self._item.to("cuda:0", use_half=True, **self.annotator_kwargs)
        return self._item

    def cleanup(self) -> None:
        if self.need_change_device and torch.cuda.is_available():
            self._item.to("cpu", use_half=False, **self.annotator_kwargs)
            torch.cuda.empty_cache()

    def unload(self) -> None:
        self.cleanup()
        return super().unload()


class APIPool(ILoadablePool[IAPI]):
    def register(self, key: str, init_fn: APIInit) -> None:
        def _init(init: bool) -> LoadableAPI:
            kw = dict(
                force_not_lazy=key in (APIs.SD, APIs.SD_INPAINTING),
                has_annotator=key in (APIs.SD, APIs.SD_INPAINTING),
            )
            api = LoadableAPI(init_fn, init=False, **kw)
            if init:
                print("> init", key, "(lazy)" if api.lazy else "")
                api.load(no_change=api.lazy)
            return api

        if key in self:
            return
        return super().register(key, _init)

    def cleanup(self, key: str) -> None:
        loadable_api: Optional[LoadableAPI] = self.pool.get(key)
        if loadable_api is None:
            raise ValueError(f"key '{key}' does not exist")
        loadable_api.cleanup()

    def need_change_device(self, key: str) -> bool:
        loadable_api: Optional[LoadableAPI] = self.pool.get(key)
        if loadable_api is None:
            raise ValueError(f"key '{key}' does not exist")
        return loadable_api.need_change_device

    def update_limit(self) -> None:
        self.limit = pool_limit()


api_pool = APIPool()
