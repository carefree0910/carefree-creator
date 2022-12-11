import torch

import numpy as np

from abc import ABCMeta
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Callable
from typing import Optional
from pydantic import Field
from pydantic import BaseModel
from functools import partial
from cfclient.models import TextModel
from cfclient.models import ImageModel
from cfclient.models import AlgorithmBase
from cfcv.misc.toolkit import np_to_bytes
from cflearn.api.cv import DiffusionAPI
from cflearn.api.cv import TranslatorAPI

from .parameters import OPT
from .parameters import verbose
from .parameters import save_gpu_ram


apis = {}
init_fns = {}
api_type = Union[DiffusionAPI, TranslatorAPI]


class SDVersions(str, Enum):
    v1_5_BC = ""
    v1_5 = "v1.5"
    ANIME = "anime"
    ANIME_ANYTHING = "anime_anything"


def _get(key: str, init_fn: Callable) -> api_type:
    m = apis.get(key)
    if m is not None:
        return m
    print("> init", key)
    if save_gpu_ram():
        m = init_fn("cpu")
    else:
        m = init_fn("cuda:0", use_half=True)
    apis[key] = m
    init_fns[key] = init_fn
    return m


def get_sd_version(version: SDVersions) -> DiffusionAPI:
    return _get(f"sd_{version}", partial(DiffusionAPI.from_sd_version, version))


def get_sd_anime() -> DiffusionAPI:
    return _get("sd_anime", DiffusionAPI.from_sd_anime)


def get_sd_inpainting() -> DiffusionAPI:
    return _get("sd_inpainting", DiffusionAPI.from_sd_inpainting)


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


# API models


class CallbackModel(BaseModel):
    callback_url: str = Field("", description="callback url to post to")


class MaxWHModel(BaseModel):
    max_wh: int = Field(1024, description="The maximum resolution.")


class VariationModel(BaseModel):
    seed: int = Field(..., description="Seed of the variation.")
    strength: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Strength of the variation.",
    )


class SDSamplers(str, Enum):
    DDIM = "ddim"
    PLMS = "plms"
    KLMS = "klms"
    SOLVER = "solver"
    K_EULER = "k_euler"
    K_EULER_A = "k_euler_a"
    K_HEUN = "k_heun"


class DiffusionModel(CallbackModel):
    use_circular: bool = Field(
        False,
        description="Whether should we use circular pattern (e.g. generate textures).",
    )
    seed: int = Field(
        -1,
        ge=-1,
        lt=2**32,
        description="""
Seed of the generation.
> If `-1`, then random seed will be used.
""",
    )
    variation_seed: int = Field(
        0,
        ge=0,
        lt=2**32,
        description="""
Seed of the variation generation.
> Only take effects when `variation_strength` is larger than 0.
""",
    )
    variation_strength: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Strength of the variation generation.",
    )
    variations: List[VariationModel] = Field([], description="Variation ingredients")
    num_steps: int = Field(20, description="Number of sampling steps", ge=5, le=100)
    guidance_scale: float = Field(
        7.5,
        description="Guidance scale for classifier-free guidance.",
    )
    negative_prompt: str = Field(
        "",
        description="Negative prompt for classifier-free guidance.",
    )
    is_anime: bool = Field(
        False,
        description="Whether should we generate anime images or not.",
    )
    version: SDVersions = Field(
        SDVersions.v1_5,
        description="Version of the diffusion model",
    )
    sampler: SDSamplers = Field(
        SDSamplers.K_EULER,
        description="Sampler of the diffusion model",
    )
    clip_skip: int = Field(
        -1,
        ge=-1,
        le=8,
        description="""
Number of CLIP layers that we want to skip.
> If it is set to `-1`, then `clip_skip` = 1 if `is_anime` else 0.
""",
    )
    custom_embeddings: Dict[str, List[List[float]]] = Field(
        {},
        description="Custom embeddings, often used in textual inversion.",
    )


class Txt2ImgModel(TextModel, MaxWHModel, DiffusionModel):
    pass


class Img2ImgModel(ImageModel, MaxWHModel):
    pass


class Img2ImgDiffusionModel(Img2ImgModel, DiffusionModel):
    pass


def handle_diffusion_model(m: DiffusionAPI, data: DiffusionModel) -> Dict[str, Any]:
    seed = None if data.seed == -1 else data.seed
    variation_seed = None
    variation_strength = None
    if data.variation_strength > 0:
        variation_seed = data.variation_seed
        variation_strength = data.variation_strength
    if data.variations is None:
        variations = None
    else:
        variations = [(v.seed, v.strength) for v in data.variations]
    m.switch_circular(data.use_circular)
    unconditional_cond = [data.negative_prompt] if data.negative_prompt else None
    clip_skip = data.clip_skip
    if clip_skip == -1:
        if data.is_anime or data.version in (
            SDVersions.ANIME,
            SDVersions.ANIME_ANYTHING,
        ):
            clip_skip = 1
        else:
            clip_skip = 0
    return dict(
        seed=seed,
        variation_seed=variation_seed,
        variation_strength=variation_strength,
        variations=variations,
        num_steps=data.num_steps,
        unconditional_guidance_scale=data.guidance_scale,
        unconditional_cond=unconditional_cond,
        sampler=data.sampler,
        verbose=verbose(),
        clip_skip=clip_skip,
        custom_embeddings=data.custom_embeddings or None,
    )


class GetPromptModel(BaseModel):
    text: str
    need_translate: bool = Field(
        True,
        description="Whether we need to translate the input text.",
    )


class GetPromptResponse(BaseModel):
    text: str
    success: bool
    reason: str


def endpoint2algorithm(endpoint: str) -> str:
    return endpoint[1:].replace("/", ".")


class IAlgorithm(AlgorithmBase, metaclass=ABCMeta):
    model_class: Type[BaseModel]
    last_latencies: Dict[str, float] = {}

    @classmethod
    def auto_register(cls) -> Callable[[AlgorithmBase], AlgorithmBase]:
        def _register(cls_: AlgorithmBase) -> AlgorithmBase:
            return cls.register(endpoint2algorithm(cls_.endpoint))(cls_)

        return _register

    def log_times(self, latencies: Dict[str, float]) -> None:
        super().log_times(latencies)
        self.last_latencies = latencies


# kafka


class Status(str, Enum):
    PENDING = "pending"
    WORKING = "working"
    FINISHED = "finished"
    EXCEPTION = "exception"
    INTERRUPTED = "interrupted"
    NOT_FOUND = "not_found"


# shortcuts


class SDParameters(BaseModel):
    is_anime: bool
    version: SDVersions


def init_sd_ms() -> Dict[str, DiffusionAPI]:
    focus = OPT.get("focus", "all")
    ms = {}
    if focus in ("all", "sd", "sd.base"):
        ms.update(
            {
                "": get_sd_version("v1.5"),
                "v1.5": get_sd_version("v1.5"),
            }
        )
    if focus in ("all", "sd", "sd.anime"):
        ms["anime"] = get_sd_anime()
        ms["anime_anything"] = get_sd_version("anime_anything_v3")
    return ms


def get_sd_from(ms: Dict[str, DiffusionAPI], data: SDParameters) -> DiffusionAPI:
    if not data.is_anime:
        m = ms[data.version]
    else:
        if data.version == SDVersions.ANIME_ANYTHING:
            m = ms["anime_anything"]
        else:
            m = ms["anime"]
    if save_gpu_ram():
        m.to("cuda:0", use_half=True)
    return m


def cleanup(m: DiffusionAPI) -> None:
    if save_gpu_ram():
        m.to("cpu")
        torch.cuda.empty_cache()


def get_api(key: str) -> Optional[api_type]:
    return apis.get(key)


def get_init_fn(key: str) -> Optional[Callable]:
    return init_fns.get(key)


def available_apis() -> List[str]:
    return sorted(apis)


__all__ = [
    "endpoint2algorithm",
    "DiffusionModel",
    "Txt2ImgModel",
    "Img2ImgModel",
    "Img2ImgDiffusionModel",
    "GetPromptModel",
    "GetPromptResponse",
    "Status",
    "IAlgorithm",
    "get_api",
    "get_init_fn",
    "available_apis",
]
