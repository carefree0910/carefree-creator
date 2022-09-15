from typing import Callable
from pydantic import Field
from pydantic import BaseModel
from cfclient.models import TextModel
from cflearn.api.cv import DiffusionAPI


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


class MaxWHModel(BaseModel):
    max_wh: int = Field(512, description="The maximum resolution.")


class Txt2ImgModel(TextModel, MaxWHModel):
    w: int = Field(512, description="The desired output width.")
    h: int = Field(512, description="The desired output height.")


__all__ = [
    "Txt2ImgModel",
]
