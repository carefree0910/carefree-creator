import numpy as np

from typing import Callable
from pydantic import Field
from pydantic import BaseModel
from cfclient.models import TextModel
from cfclient.models import ImageModel
from cfcv.misc.toolkit import np_to_bytes
from cflearn.api.cv import DiffusionAPI
from cflearn.api.cv import TranslatorAPI


apis = {}


def _get(key: str, init: Callable) -> DiffusionAPI:
    m = apis.get(key)
    if m is not None:
        return m
    m = init("cuda:0", use_amp=True)
    apis[key] = m
    return m


def get_sd() -> DiffusionAPI:
    return _get("sd", DiffusionAPI.from_sd)


def get_sr() -> TranslatorAPI:
    return _get("sr", TranslatorAPI.from_esr)


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


class Img2ImgModel(ImageModel, MaxWHModel):
    pass


__all__ = [
    "Txt2ImgModel",
    "Img2ImgModel",
]
