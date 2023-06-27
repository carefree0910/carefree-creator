import os
import torch

import numpy as np

from abc import ABCMeta
from PIL import Image
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import TypeVar
from typing import Callable
from typing import Optional
from fastapi import Response
from pathlib import Path
from pydantic import Field
from pydantic import BaseModel
from functools import partial
from cftool.cv import np_to_bytes
from cftool.misc import shallow_copy_dict
from cftool.types import TNumberPair
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
from cflearn.api.cv.diffusion import InpaintingMode
from cflearn.api.cv.diffusion import InpaintingSettings
from cflearn.misc.toolkit import download_model
from cflearn.models.cv.diffusion import StableDiffusion
from cflearn.api.cv.third_party.blip import BLIPAPI
from cflearn.api.cv.third_party.lama import LaMa
from cflearn.api.cv.third_party.isnet import ISNetAPI
from cflearn.api.cv.third_party.prompt import PromptEnhanceAPI

from .cos import download_with_retry
from .cos import download_image_with_retry
from .utils import api_pool
from .utils import to_canvas
from .utils import APIs
from .parameters import verbose
from .parameters import get_focus
from .parameters import pool_limit
from .parameters import Focus


class SDInpaintingVersions(str, Enum):
    v1_5 = "v1.5"


BaseSDTag = "_base_sd"
NUM_CONTROL_POOL = 1


def _base_sd_path() -> str:
    root = os.path.join(OPT.cache_dir, DLZoo.model_dir)
    return download_model("ldm_sd_v1.5", root=root)


def _get(init_fn: Callable, init_to_cpu: bool) -> Any:
    if init_to_cpu:
        return init_fn()
    return init_fn("cuda:0", use_half=True)


def init_sd(init_to_cpu: bool) -> ControlledDiffusionAPI:
    version = SDVersions.v1_5
    kw = dict(num_pool=NUM_CONTROL_POOL, lazy=True)
    init_fn = partial(ControlledDiffusionAPI.from_sd_version, version, **kw)
    m: ControlledDiffusionAPI = _get(init_fn, init_to_cpu)
    focus = get_focus()
    if focus != Focus.SYNC:
        m.sd_weights.limit = pool_limit()
        m.current_sd_version = version
        print("> registering base sd")
        m.prepare_sd([version])
        m.sd_weights.register(BaseSDTag, _base_sd_path())
        print("> warmup ControlNet")
        m.switch_control(*m.preset_control_hints)
    print("> prepare ControlNet Annotators")
    m.prepare_annotators()
    return m


def init_sd_inpainting(init_to_cpu: bool) -> ControlledDiffusionAPI:
    kw = dict(num_pool=NUM_CONTROL_POOL, lazy=True)
    init_fn = partial(ControlledDiffusionAPI.from_sd_inpainting, **kw)
    api: ControlledDiffusionAPI = _get(init_fn, init_to_cpu)
    # manually maintain sd_weights
    ## original weights
    api.sd_weights.register(BaseSDTag, _base_sd_path())
    ## inpainting weights
    root = os.path.join(OPT.cache_dir, DLZoo.model_dir)
    inpainting_path = download_model("ldm.sd_inpainting", root=root)
    api.sd_weights.register(SDInpaintingVersions.v1_5, inpainting_path)
    api.current_sd_version = SDInpaintingVersions.v1_5
    # inject properties from sd
    register_sd()
    sd: ControlledDiffusionAPI = api_pool.get(APIs.SD, no_change=True)
    api.annotators = sd.annotators
    api.controlnet_weights = sd.controlnet_weights
    api.switch_control(*api.preset_control_hints)
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


def register_esr_ultrasharp() -> None:
    def _init(*args: Any, **kw: Any) -> TranslatorAPI:
        m = TranslatorAPI.from_esr(*args, **kw)
        sr_folder = os.path.join(OPT.external_dir, "sr")
        model_path = os.path.join(sr_folder, "4x-UltraSharp.ckpt")
        if not os.path.isfile(model_path):
            raise ValueError(f"cannot find {model_path}")
        m.m.load_state_dict(torch.load(model_path))
        return m

    api_pool.register(
        APIs.ESR_ULTRASHARP,
        lambda init_to_cpu: _get(_init, init_to_cpu),
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
    custom_embeddings: Dict[str, Optional[List[List[float]]]] = Field(
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


class CommonSDInpaintingModel(ReturnArraysModel, MaxWHModel):
    keep_original: bool = Field(
        False,
        description="Whether strictly keep the original image identical in the output image.",
    )
    use_raw_inpainting: bool = Field(
        False,
        description="""
Whether use the raw inpainting method.
> This is useful when you want to apply inpainting with custom SD models.
""",
    )
    use_background_guidance: bool = Field(
        False,
        description="""
Whether inject the latent of the background during the generation.
> If `use_raw_inpainting`, this will always be `True` because in this case
the latent of the background is the only information for us to inpaint.
""",
    )
    use_reference: bool = Field(
        False,
        description="Whether use the original image as reference.",
    )
    reference_fidelity: float = Field(
        0.0,
        description="Fidelity of the reference image, only take effects when `use_reference` is `True`.",
    )
    inpainting_mode: InpaintingMode = Field(
        InpaintingMode.NORMAL,
        description="Inpainting mode. MASKED is preferred when the masked area is small.",
    )
    inpainting_mask_blur: Optional[int] = Field(
        None,
        description="The smoothness of the inpainting's mask, `None` means no smooth.",
    )
    inpainting_mask_padding: Optional[int] = Field(
        32,
        description="Padding of the inpainting mask under MASKED mode. If `None`, then no padding.",
    )
    inpainting_mask_binary_threshold: Optional[int] = Field(
        32,
        description="Binary threshold of the inpainting mask under MASKED mode. If `None`, then no thresholding.",
    )
    inpainting_target_wh: TNumberPair = Field(
        None,
        description="Target width and height of the images under MASKED mode.",
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


class _ControlNetCoreModel(BaseModel):
    hint_url: str = Field(
        "",
        description="""
The `cdn` / `cos` url of the user's hint image.
> If empty string is provided, we will use `url` as `hint_url`.
> `cos` url from `qcloud` is preferred.
""",
    )
    hint_annotator: Optional[str] = Field(
        None,
        description="""
The annotator type of the hint.
> If not specified, will use the control type as the annotator's type.
""",
    )
    hint_start: Optional[float] = Field(None, description="start ratio of the control")
    bypass_annotator: bool = Field(False, description="Bypass the annotator.")
    guess_mode: bool = Field(False, description="Guess mode.")
    no_switch: bool = Field(
        False,
        description="Whether not to switch the ControlNet weights even when the base model has switched.",
    )


class _ControlNetModel(_ControlNetCoreModel):
    url: Optional[str] = Field(None, description="specify this to do img2img")
    prompt: str = Field("", description="Prompt.")
    fidelity: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="The fidelity of the input image, only take effects when `url` is not `None`.",
    )
    num_samples: int = Field(1, ge=1, le=4, description="Number of samples.")
    base_model: str = Field(
        SDVersions.v1_5,
        description="The base model.",
    )
    use_audit: bool = Field(False, description="Whether audit the outputs.")
    mask_url: Optional[str] = Field(None, description="specify this to do inpainting")
    use_inpainting: bool = Field(False, description="Whether use inpainting model.")

    @property
    def version(self) -> str:
        return self.base_model


class ControlNetModel(CommonSDInpaintingModel, DiffusionModel, _ControlNetModel):
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
            user_folder = os.path.expanduser("~")
            external_folder = os.path.join(user_folder, ".cache", "external")
            lora_folder = os.path.join(external_folder, "lora")
            for key in data.lora_scales:
                if model.lora_manager.has(key):
                    continue
                if not os.path.isdir(lora_folder):
                    raise ValueError(
                        f"'{key}' does not exist in current loaded lora "
                        f"and '{lora_folder}' does not exist either."
                    )
                for lora_file in os.listdir(lora_folder):
                    lora_name = os.path.splitext(lora_file)[0]
                    if key != lora_name:
                        continue
                    try:
                        print(f">> loading {key}")
                        lora_path = os.path.join(lora_folder, lora_file)
                        m.load_sd_lora(lora_name, path=lora_path)
                    except Exception as err:
                        raise ValueError(f"failed to load {key}: {err}")
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
        anchor=64,
        max_wh=data.max_wh,
        keep_original=data.keep_original,
        use_raw_inpainting=data.use_raw_inpainting,
        use_background_guidance=data.use_background_guidance,
        use_reference=data.use_reference,
        reference_fidelity=data.reference_fidelity,
        inpainting_settings=InpaintingSettings(
            data.inpainting_mode,
            data.inpainting_mask_blur,
            data.inpainting_mask_padding,
            data.inpainting_mask_binary_threshold,
            data.inpainting_target_wh,
        ),
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


TAlgo = TypeVar("TAlgo", bound=Type[AlgorithmBase])


class IAlgorithm(AlgorithmBase, metaclass=ABCMeta):
    model_class: Type[BaseModel]
    response_model_class: Optional[Type[BaseModel]] = None
    last_latencies: Dict[str, float] = {}

    @classmethod
    def auto_register(cls) -> Callable[[TAlgo], TAlgo]:
        def _register(cls_: TAlgo) -> TAlgo:
            return cls.register(endpoint2algorithm(cls_.endpoint))(cls_)

        return _register

    def log_times(self, latencies: Dict[str, float]) -> None:
        from cfcreator.sdks.apis import ALL_LATENCIES_KEY

        self.last_latencies = shallow_copy_dict(latencies)
        latencies.pop(ALL_LATENCIES_KEY, None)
        super().log_times(latencies)

    async def download_with_retry(self, url: str) -> bytes:
        return await download_with_retry(self.http_client.session, url)

    async def download_image_with_retry(self, url: str) -> Image.Image:
        return await download_image_with_retry(self.http_client.session, url)

    async def get_image_from(
        self,
        key: str,
        data: BaseModel,
        kwargs: Dict[str, Any],
    ) -> Image.Image:
        existing = kwargs.pop(key, None)
        if existing is not None and isinstance(existing, Image.Image):
            return existing
        return await self.download_image_with_retry(getattr(data, key))


class IWrapperAlgorithm(IAlgorithm):
    algorithms: Optional[Dict[str, IAlgorithm]] = None

    def initialize(self) -> None:
        from cfcreator.sdks.apis import APIs
        from cfcreator.sdks.apis import ALL_LATENCIES_KEY

        if self.algorithms is None:
            raise ValueError("`algorithms` should be provided for `IWrapperAlgorithm`.")
        self.apis = APIs(
            clients=self.clients,
            algorithms=self.algorithms,
            verbose=None,
            lazy_load=None,
        )
        self.latencies_key = ALL_LATENCIES_KEY


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


def get_response(data: ReturnArraysModel, results: List[np.ndarray]) -> Any:
    if data.return_arrays:
        return results
    return Response(content=np_to_bytes(to_canvas(results)), media_type="image/png")


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
    "IWrapperAlgorithm",
]
