import numpy as np

from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from pydantic import Field
from pydantic import BaseModel
from cfclient.models import TextModel
from cfclient.models import ImageModel
from cfcv.misc.toolkit import np_to_bytes
from cflearn.api.cv import DiffusionAPI
from cflearn.api.cv import TranslatorAPI


apis = {}


def _get(key: str, init: Callable) -> Union[DiffusionAPI, TranslatorAPI]:
    m = apis.get(key)
    if m is not None:
        return m
    m = init("cuda:0", use_half=True)
    apis[key] = m
    return m


def get_sd() -> DiffusionAPI:
    return _get("sd", DiffusionAPI.from_sd)


def get_esr() -> TranslatorAPI:
    return _get("esr", TranslatorAPI.from_esr)


def get_esr_anime() -> TranslatorAPI:
    return _get("esr_anime", TranslatorAPI.from_esr_anime)


def get_inpainting() -> DiffusionAPI:
    return _get("inpainting", DiffusionAPI.from_inpainting)


def get_semantic() -> DiffusionAPI:
    return _get("semantic", DiffusionAPI.from_semantic)


def get_bytes_from_translator(img_arr: np.ndarray) -> bytes:
    img_arr = img_arr.transpose([1, 2, 0])
    return np_to_bytes(img_arr)


def get_bytes_from_diffusion(img_arr: np.ndarray) -> bytes:
    img_arr = 0.5 * (img_arr + 1.0)
    img_arr = img_arr.transpose([1, 2, 0])
    return np_to_bytes(img_arr)


class MaxWHModel(BaseModel):
    max_wh: int = Field(512, description="The maximum resolution.")


class Txt2ImgModel(TextModel, MaxWHModel):
    w: int = Field(512, description="The desired output width.")
    h: int = Field(512, description="The desired output height.")
    use_circular: bool = Field(
        False,
        description="Whether should we use circular pattern (e.g. generate textures).",
    )
    use_seed: bool = Field(False, description="Whether should we use seed.")
    seed: int = Field(0, description="""
Seed of the generation.
> Only take effects when `use_refine` is set to True.
"""
    )
    variation_seed: int = Field(0, description="""
Seed of the variation generation.
> Only take effects when `variation_strength` is larger than 0.
"""
    )
    variation_strength: float = Field(
        0.0,
        ge=0.0,
        description="Strength of the variation generation.",
    )
    variations: List[Tuple[int, float]] = Field([], description="Variation ingredients")


class Img2ImgModel(ImageModel, MaxWHModel):
    pass


__all__ = [
    "Txt2ImgModel",
    "Img2ImgModel",
]
