from typing import Callable
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


class Txt2ImgModel(TextModel):
    pass


__all__ = [
    "Txt2ImgModel",
]
