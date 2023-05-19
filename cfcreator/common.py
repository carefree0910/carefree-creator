import os
import json
import torch
import cflearn

import numpy as np

from abc import ABCMeta
from PIL import Image
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Callable
from typing import Optional
from pathlib import Path
from pydantic import Field
from pydantic import BaseModel
from functools import partial
from cftool.cv import np_to_bytes
from cflearn.zoo import DLZoo
from cflearn.parameters import OPT
from cfclient.models import TextModel
from cfclient.models import ImageModel
from cfclient.models import AlgorithmBase
from cflearn.api.cv import SDVersions
from cflearn.api.cv import DiffusionAPI
from cflearn.api.cv import TranslatorAPI
from cflearn.api.cv import ImageHarmonizationAPI
from cflearn.api.cv import ControlledDiffusionAPI
from cflearn.misc.toolkit import download_model
from cflearn.models.cv.diffusion import StableDiffusion
from cflearn.api.cv.third_party.blip import BLIPAPI
from cflearn.api.cv.third_party.lama import LaMa
from cflearn.api.cv.third_party.isnet import ISNetAPI
from cflearn.api.cv.third_party.prompt import PromptEnhanceAPI

from .cos import download_with_retry
from .cos import download_image_with_retry
from .utils import api_pool
from .utils import APIs
from .parameters import verbose
from .parameters import get_focus
from .parameters import pool_limit
from .parameters import Focus


class SDInpaintingVersions(str, Enum):
    v1_5 = "v1.5"


BaseSDTag = "_base_sd"


def _base_sd_path() -> str:
    root = os.path.join(OPT.cache_dir, DLZoo.model_dir)
    return download_model("ldm_sd_v1.5", root=root)


def _get(init_fn: Callable, init_to_cpu: bool) -> Any:
    if init_to_cpu:
        return init_fn()
    return init_fn("cuda:0", use_half=True)


def _load_lora(m: ControlledDiffusionAPI, external_folder: str) -> None:
    print("> loading lora")
    num_lora = 0
    lora_folder = os.path.join(external_folder, "lora")
    os.makedirs(lora_folder, exist_ok=True)
    for lora_file in os.listdir(lora_folder):
        try:
            lora_name = os.path.splitext(lora_file)[0]
            lora_path = os.path.join(lora_folder, lora_file)
            print(f">> loading {lora_name}")
            m.load_sd_lora(lora_name, path=lora_path)
            num_lora += 1
        except:
            print(f">>>> Failed to load!")
            continue
    print(f"> {num_lora} lora loaded")


def init_sd(init_to_cpu: bool) -> ControlledDiffusionAPI:
    version = SDVersions.v1_5
    init_fn = partial(ControlledDiffusionAPI.from_sd_version, version)
    m: ControlledDiffusionAPI = _get(init_fn, init_to_cpu)
    focus = get_focus()
    m.sd_weights.limit = pool_limit()
    m.current_sd_version = version
    print("> registering base sd")
    m.prepare_sd([version])
    m.sd_weights.register(BaseSDTag, _base_sd_path())
    # when focus is SYNC, `init_sd` is called because we need to expose `control_hint`
    # endpoints. However, `sd` itself will never be used, so we can skip some stuffs
    user_folder = os.path.expanduser("~")
    external_folder = os.path.join(user_folder, ".cache", "external")
    if focus == Focus.SYNC:
        print("> prepare ControlNet Annotators")
        for hint in m.control_defaults:
            m.prepare_annotator(hint)
    else:
        print("> prepare ControlNet weights")
        m.prepare_control_defaults()
        print("> prepare ControlNet Annotators")
        m.prepare_annotators()
        print("> warmup ControlNet")
        m.switch_control(*m.available_control_hints)
    # lora stuffs
    _load_lora(m, external_folder)
    # return
    return m


def init_sd_inpainting(init_to_cpu: bool) -> ControlledDiffusionAPI:
    register_sd()
    sd: ControlledDiffusionAPI = api_pool.get(APIs.SD, no_change=True)
    init_fn = ControlledDiffusionAPI.from_sd_inpainting
    api: ControlledDiffusionAPI = _get(init_fn, init_to_cpu)
    # manually maintain sd_weights
    ## original weights
    api.sd_weights.register(BaseSDTag, _base_sd_path())
    ## inpainting weights
    root = os.path.join(OPT.cache_dir, DLZoo.model_dir)
    inpainting_path = download_model("ldm.sd_inpainting", root=root)
    api.sd_weights.register(SDInpaintingVersions.v1_5, inpainting_path)
    api.current_sd_version = SDInpaintingVersions.v1_5
    # lora stuffs
    user_folder = os.path.expanduser("~")
    external_folder = os.path.join(user_folder, ".cache", "external")
    _load_lora(api, external_folder)
    # inject properties from sd
    api.annotators = sd.annotators
    api.controlnet_weights = sd.controlnet_weights
    api.switch_control(*api.available_control_hints)
    return api


def register_sd() -> None:
    api_pool.register(APIs.SD, init_sd)


def register_sd_inpainting() -> None:
    api_pool.register(APIs.SD_INPAINTING, init_sd_inpainting)


def register_esr() -> None:
    api_pool.register(
        APIs.ESR,
        lambda init_to_cpu: _get(TranslatorAPI.from_esr, init_to_cpu),
    )


def register_esr_anime() -> None:
    api_pool.register(
        APIs.ESR_ANIME,
        lambda init_to_cpu: _get(TranslatorAPI.from_esr_anime, init_to_cpu),
    )


def register_inpainting() -> None:
    api_pool.register(
        APIs.INPAINTING,
        lambda init_to_cpu: _get(DiffusionAPI.from_inpainting, init_to_cpu),
    )


def register_lama() -> None:
    api_pool.register(
        APIs.LAMA,
        lambda init_to_cpu: _get(LaMa, init_to_cpu),
    )


def register_semantic() -> None:
    api_pool.register(
        APIs.SEMANTIC,
        lambda init_to_cpu: _get(DiffusionAPI.from_semantic, init_to_cpu),
    )


def register_hrnet() -> None:
    api_pool.register(
        APIs.HRNET,
        lambda init_to_cpu: _get(ImageHarmonizationAPI, init_to_cpu),
    )


def register_isnet() -> None:
    api_pool.register(
        APIs.ISNET,
        lambda init_to_cpu: _get(ISNetAPI, init_to_cpu),
    )


def register_blip() -> None:
    api_pool.register(
        APIs.BLIP,
        lambda init_to_cpu: _get(BLIPAPI, init_to_cpu),
    )


def register_prompt_enhance() -> None:
    api_pool.register(
        APIs.PROMPT_ENHANCE,
        lambda init_to_cpu: _get(PromptEnhanceAPI, init_to_cpu),
    )


def get_bytes_from_translator(img_arr: np.ndarray, *, transpose: bool = True) -> bytes:
    if transpose:
        img_arr = img_arr.transpose([1, 2, 0])
    return np_to_bytes(img_arr)


def get_normalized_arr_from_diffusion(img_arr: np.ndarray) -> np.ndarray:
    img_arr = 0.5 * (img_arr + 1.0)
    img_arr = img_arr.transpose([1, 2, 0])
    return img_arr


def get_bytes_from_diffusion(img_arr: np.ndarray) -> bytes:
    return np_to_bytes(get_normalized_arr_from_diffusion(img_arr))


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


class TomeInfoModel(BaseModel):
    enable: bool = Field(False, description="Whether enable tomesd.")
    ratio: float = Field(0.5, description="The ratio of tokens to merge.")
    max_downsample: int = Field(
        1,
        description="Apply ToMe to layers with at most this amount of downsampling.",
    )
    sx: int = Field(2, description="The stride for computing dst sets.")
    sy: int = Field(2, description="The stride for computing dst sets.")
    seed: int = Field(
        -1,
        ge=-1,
        lt=2**32,
        description="""
Seed of the generation.
> If `-1`, then seed from `DiffusionModel` will be used.
> If `DiffusionModel.seed` is also `-1`, then random seed will be used.
""",
    )
    use_rand: bool = Field(True, description="Whether allow random perturbations.")
    merge_attn: bool = Field(True, description="Whether merge attention.")
    merge_crossattn: bool = Field(False, description="Whether merge cross attention.")
    merge_mlp: bool = Field(False, description="Whether merge mlp.")


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
    variations: List[VariationModel] = Field(
        default_factory=lambda: [],
        description="Variation ingredients",
    )
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
    version: str = Field(
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
    tome_info: TomeInfoModel = Field(TomeInfoModel(), description="tomesd settings.")
    lora_scales: Optional[Dict[str, float]] = Field(
        None,
        description="lora scales, key is the name, value is the weight.",
    )
    lora_paths: Optional[List[str]] = Field(
        None,
        description="If provided, we will dynamically load lora from the given paths.",
    )


class ReturnArraysModel(BaseModel):
    return_arrays: bool = Field(
        False,
        description="Whether return List[np.ndarray] directly, only for internal usages.",
    )


class CommonSDInpaintingModel(ReturnArraysModel):
    keep_original: bool = Field(
        False,
        description="Whether strictly keep the original image identical in the output image.",
    )
    ref_mask_smooth: int = Field(
        3,
        description="The smoothness of the reference's mask, `0` means no smooth.",
    )
    use_raw_inpainting: bool = Field(
        False,
        description="""
Whether use the raw inpainting method.
> This is useful when you want to apply inpainting with custom SD models.
""",
    )
    raw_inpainting_use_ref: bool = Field(
        False,
        description="Whether use input image as reference.",
    )
    raw_inpainting_fidelity: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="The fidelity of the input image when it is used as reference in raw inpainting.",
    )
    use_latent_guidance: bool = Field(
        False,
        description="Whether use the latent of the givent image to guide the generation.",
    )
    reference_fidelity: float = Field(
        0.0, description="Fidelity of the reference image."
    )


class HighresModel(BaseModel):
    fidelity: float = Field(0.3, description="Fidelity of the original latent.")
    upscale_factor: float = Field(2.0, description="Upscale factor.")
    max_wh: int = Field(1024, description="Max width or height of the output image.")


class Txt2ImgModel(DiffusionModel, MaxWHModel, TextModel):
    pass


class Img2ImgModel(MaxWHModel, ImageModel):
    pass


class Img2ImgDiffusionModel(DiffusionModel, Img2ImgModel):
    pass


class ControlStrengthModel(BaseModel):
    control_strength: float = Field(1.0, description="The strength of the control.")


class _ControlNetModel(BaseModel):
    url: Optional[str] = Field(None, description="specify this to do img2img")
    hint_url: str = Field(
        "",
        description="""
The `cdn` / `cos` url of the user's hint image.
> If empty string is provided, we will use `url` as `hint_url`.
> `cos` url from `qcloud` is preferred.
""",
    )
    hint_start: Optional[float] = Field(None, description="start ratio of the control")
    prompt: str = Field("", description="Prompt.")
    fidelity: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="The fidelity of the input image, only take effects when `url` is not `None`.",
    )
    num_samples: int = Field(1, ge=1, le=4, description="Number of samples.")
    bypass_annotator: bool = Field(False, description="Bypass the annotator.")
    base_model: str = Field(
        SDVersions.v1_5,
        description="The base model.",
    )
    guess_mode: bool = Field(False, description="Guess mode.")
    use_audit: bool = Field(False, description="Whether audit the outputs.")
    no_switch: bool = Field(
        False,
        description="Whether not to switch the ControlNet weights even when the base model has switched.",
    )
    mask_url: Optional[str] = Field(None, description="specify this to do inpainting")

    @property
    def version(self) -> str:
        return self.base_model


class ControlNetModel(
    CommonSDInpaintingModel,
    DiffusionModel,
    MaxWHModel,
    _ControlNetModel,
):
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
        if data.is_anime or data.version.startswith("anime"):
            clip_skip = 1
        else:
            clip_skip = 0
    tome_info = data.tome_info.dict()
    enable_tome = tome_info.pop("enable")
    if not enable_tome:
        m.set_tome_info(None)
    else:
        if tome_info["seed"] == -1:
            if seed is None:
                tome_info.pop("seed")
            else:
                tome_info["seed"] = seed
        m.set_tome_info(tome_info)
    # lora
    model = m.m
    if isinstance(model, StableDiffusion):
        manager = model.lora_manager
        if manager.injected:
            m.cleanup_sd_lora()
        if data.lora_scales is not None:
            m.inject_sd_lora(*list(data.lora_scales))
            m.set_sd_lora_scales(data.lora_scales)
    # return
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


def handle_diffusion_inpainting_model(data: CommonSDInpaintingModel) -> Dict[str, Any]:
    return dict(
        ref_mask_smooth=data.ref_mask_smooth,
        use_raw_inpainting=data.use_raw_inpainting,
        raw_inpainting_use_ref=data.raw_inpainting_use_ref,
        raw_inpainting_fidelity=data.raw_inpainting_fidelity,
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
    response_model_class: Optional[Type[BaseModel]] = None
    last_latencies: Dict[str, float] = {}

    @classmethod
    def auto_register(cls) -> Callable[[AlgorithmBase], AlgorithmBase]:
        def _register(cls_: AlgorithmBase) -> AlgorithmBase:
            return cls.register(endpoint2algorithm(cls_.endpoint))(cls_)

        return _register

    def log_times(self, latencies: Dict[str, float]) -> None:
        super().log_times(latencies)
        self.last_latencies = latencies

    async def download_with_retry(self, url: str) -> bytes:
        return await download_with_retry(self.http_client.session, url)

    async def download_image_with_retry(self, url: str) -> Image.Image:
        return await download_image_with_retry(self.http_client.session, url)


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
    version: str
    lora_paths: Optional[List[str]]


def load_sd_lora_with(sd: ControlledDiffusionAPI, data: SDParameters) -> None:
    if data.lora_paths is not None:
        for lora_path in data.lora_paths:
            key = Path(lora_path).stem
            sd.load_sd_lora(key, path=lora_path)


def get_sd_from(api_key: APIs, data: SDParameters, **kw: Any) -> ControlledDiffusionAPI:
    if not data.is_anime:
        version = data.version
    else:
        version = data.version if data.version.startswith("anime") else "anime"
    sd: ControlledDiffusionAPI = api_pool.get(api_key, **kw)
    if api_key != APIs.SD_INPAINTING:
        sd.prepare_sd([version])
    elif version != SDInpaintingVersions.v1_5:
        sd.prepare_sd([version], sub_folder="inpainting", force_external=True)
    sd.switch_sd(version)
    sd.disable_control()
    load_sd_lora_with(sd, data)
    return sd


__all__ = [
    "endpoint2algorithm",
    "DiffusionModel",
    "HighresModel",
    "Txt2ImgModel",
    "Img2ImgModel",
    "Img2ImgDiffusionModel",
    "ReturnArraysModel",
    "ControlNetModel",
    "GetPromptModel",
    "GetPromptResponse",
    "Status",
    "IAlgorithm",
]
