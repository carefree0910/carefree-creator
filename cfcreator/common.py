import os
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
from cflearn.api.cv import SDVersions
from cflearn.api.cv import DiffusionAPI
from cflearn.api.cv import TranslatorAPI
from cflearn.api.cv import ImageHarmonizationAPI
from cflearn.api.cv import ControlledDiffusionAPI

from .cos import download_with_retry
from .cos import download_image_with_retry
from .parameters import verbose
from .parameters import get_focus
from .parameters import init_to_cpu
from .parameters import need_change_device
from .parameters import Focus


apis = {}
init_fns = {}
init_models = {}
api_type = Union[DiffusionAPI, TranslatorAPI]


class ExternalVersions(str, Enum):
    """
    Specify external SD weights that need to be loaded.
    * these weights should be placed under ~/.cache/external/ folder.
    * file name should be {version}.ckpt

    Example
    -------
    class ExternalVersions(str, Enum):
        MY_FANCY_MODEL = "my_fancy_model"

    then you can place your model at ~/.cache/external/my_fancy_model.ckpt,
    after which you can specify `my_fancy_model` as the `version` parameter!
    """


def merge_enums(*enums: Enum) -> Enum:
    members: Dict[str, str] = {}
    for e in enums:
        for name, member in e.__members__.items():
            members[name] = member.value
    return Enum("MergedVersions", members, type=str)


MergedVersions = merge_enums(SDVersions, ExternalVersions)


def _get(
    key: str,
    init_fn: Callable,
    callback: Optional[Callable] = None,
    lazy: bool = False,
) -> api_type:
    m = apis.get(key)
    if m is not None:
        return m
    print("> init", key, "(lazy)" if lazy else "")
    if init_to_cpu() or lazy:
        m = init_fn("cpu")
    else:
        m = init_fn("cuda:0", use_half=True)
    apis[key] = m
    init_fns[key] = init_fn
    if callback is not None:
        callback(m)
    return m


def _get_general_model(key: str, init_fn: Callable) -> Any:
    m = init_models.get(key)
    if m is not None:
        return m
    m = init_fn()
    init_models[key] = m
    return m


def init_sd() -> ControlledDiffusionAPI:
    def _callback(m: ControlledDiffusionAPI) -> None:
        focus = get_focus()
        m.current_sd_version = MergedVersions.v1_5
        targets = []
        common = Focus.ALL, Focus.SD, Focus.CONTROL, Focus.PIPELINE
        if focus in common + (Focus.SD_BASE,):
            targets.append(MergedVersions.v1_5)
        if focus in common + (Focus.SD_ANIME,):
            targets.append(MergedVersions.ANIME)
            targets.append(MergedVersions.DREAMLIKE)
            targets.append(MergedVersions.ANIME_ANYTHING)
            targets.append(MergedVersions.ANIME_HYBRID)
            targets.append(MergedVersions.ANIME_GUOFENG)
            targets.append(MergedVersions.ANIME_ORANGE)
        print(f"> preparing sd weights ({', '.join(targets)}) (focus={focus})")
        m.prepare_sd(targets)
        # when focus is SYNC, `init_sd` is called because we need to expose `control_hint`
        # endpoints. However, `sd` itself will never be used, so we can skip some stuffs
        if focus == Focus.SYNC:
            print("> prepare ControlNet Annotators")
            for hint in m.defaults:
                m.prepare_annotator(hint)
        else:
            print("> converting external weights")
            external_dir = os.path.join(os.path.expanduser("~"), ".cache", "external")
            for version in ExternalVersions:
                print(f">> converting {version}")
                model_path = os.path.join(external_dir, f"{version}.ckpt")
                d = cflearn.scripts.sd.convert(model_path, m, load=False)
                m.sd_weights[f"ldm_sd_{version}"] = d
            print("> prepare ControlNet weights")
            m.prepare_defaults()
            print("> prepare ControlNet Annotators")
            m.prepare_annotators()
            print("> warmup ControlNet")
            m.switch(*m.available)

    init_fn = partial(ControlledDiffusionAPI.from_sd_version, "v1.5")
    return _get("sd_v1.5", init_fn, _callback)


def get_sd_anime() -> DiffusionAPI:
    return _get("sd_anime", DiffusionAPI.from_sd_anime)


def get_sd_inpainting() -> ControlledDiffusionAPI:
    def _callback(m: ControlledDiffusionAPI) -> None:
        sd = init_sd()
        m.weights = sd.weights
        m.annotators = sd.annotators
        m.current_sd_version = MergedVersions.v1_5
        m.switch(*m.available)

    init_fn = ControlledDiffusionAPI.from_sd_inpainting
    return _get("sd_inpainting", init_fn, _callback)


def get_esr() -> TranslatorAPI:
    return _get("esr", TranslatorAPI.from_esr)


def get_esr_anime() -> TranslatorAPI:
    return _get("esr_anime", TranslatorAPI.from_esr_anime)


def get_inpainting(lazy: bool) -> DiffusionAPI:
    return _get("inpainting", DiffusionAPI.from_inpainting, lazy=lazy)


def get_semantic(lazy: bool) -> DiffusionAPI:
    return _get("semantic", DiffusionAPI.from_semantic, lazy=lazy)


def get_hrnet() -> ImageHarmonizationAPI:
    _get = lambda: ImageHarmonizationAPI("cpu" if init_to_cpu() else "cuda:0")
    return _get_general_model("hrnet", _get)


def get_bytes_from_translator(img_arr: np.ndarray) -> bytes:
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
    version: MergedVersions = Field(
        MergedVersions.v1_5,
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


class CommonSDInpaintingModel(BaseModel):
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
    raw_inpainting_fidelity: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="The fidelity of the input image when using raw inpainting.",
    )
    ref_url: str = Field(
        "",
        description="""
The `cdn` / `cos` url of the reference image.
> `cos` url from cloud is preferred.
> If empty string is provided, we will not use the reference feature.  
""",
    )
    ref_fidelity: float = Field(
        0.2,
        description="Fidelity of the reference image (if provided)",
    )


class Txt2ImgModel(TextModel, MaxWHModel, DiffusionModel):
    pass


class Img2ImgModel(ImageModel, MaxWHModel):
    pass


class Img2ImgDiffusionModel(Img2ImgModel, DiffusionModel):
    pass


class ControlStrengthModel(BaseModel):
    control_strength: float = Field(1.0, description="The strength of the control.")


class ReturnArraysModel(BaseModel):
    return_arrays: bool = Field(
        False,
        description="Whether return List[np.ndarray] directly, only for internal usages.",
    )


class ControlNetModel(DiffusionModel, MaxWHModel, ImageModel, ReturnArraysModel):
    hint_url: str = Field(
        "",
        description="""
The `cdn` / `cos` url of the user's hint image.
> If empty string is provided, we will use `url` as `hint_url`.
> `cos` url from `qcloud` is preferred.
""",
    )
    hint_starts: Dict[str, float] = Field(
        default_factory=lambda: {},
        description="start ratio of each hint",
    )
    prompt: str = Field(..., description="Prompt.")
    fidelity: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="The fidelity of the input image, only take effects when `use_img2img` is True.",
    )
    use_img2img: bool = Field(True, description="Whether use img2img method.")
    num_samples: int = Field(1, ge=1, le=4, description="Number of samples.")
    bypass_annotator: bool = Field(False, description="Bypass the annotator.")
    base_model: MergedVersions = Field(
        MergedVersions.v1_5,
        description="The base model.",
    )
    guess_mode: bool = Field(False, description="Guess mode.")
    use_audit: bool = Field(False, description="Whether audit the outputs.")


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

    async def handle_diffusion_inpainting_model(
        self,
        data: CommonSDInpaintingModel,
    ) -> Dict[str, Any]:
        if not data.ref_url:
            reference = None
        else:
            reference = await self.download_image_with_retry(data.ref_url)
        return dict(
            use_raw_inpainting=data.use_raw_inpainting,
            raw_inpainting_fidelity=data.raw_inpainting_fidelity,
            reference=reference,
            reference_fidelity=data.ref_fidelity,
        )


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
    version: MergedVersions


def get_sd_from(sd: ControlledDiffusionAPI, data: SDParameters) -> DiffusionAPI:
    if not data.is_anime:
        version = data.version
    else:
        version = data.version if data.version.startswith("anime") else "anime"
    sd.switch_sd(version)
    sd.disable_control()
    if need_change_device():
        sd.to("cuda:0", use_half=True)
    return sd


def cleanup(m: DiffusionAPI, lazy: bool = False) -> None:
    if need_change_device() or lazy:
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
    "ReturnArraysModel",
    "ControlNetModel",
    "GetPromptModel",
    "GetPromptResponse",
    "Status",
    "IAlgorithm",
    "get_api",
    "get_init_fn",
    "available_apis",
]
