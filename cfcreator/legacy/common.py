from typing import Dict
from pydantic import Field
from pydantic import BaseModel
from cfclient.models import ImageModel
from cflearn.api.cv import SDVersions

from ..common import MaxWHModel
from ..common import UseAuditModel
from ..common import DiffusionModel
from ..common import ReturnArraysModel


class _ControlNetModel(UseAuditModel):
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
    hint_ends: Dict[str, float] = Field(
        default_factory=lambda: {},
        description="end ratio of each hint",
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
    base_model: str = Field(
        SDVersions.v1_5,
        description="The base model.",
    )
    guess_mode: bool = Field(False, description="Guess mode.")
    no_switch: bool = Field(
        False,
        description="Whether not to switch the ControlNet weights even when the base model has switched.",
    )

    @property
    def version(self) -> str:
        return self.base_model


# only useful when inpainting model is used
class _InpaintingMixin(BaseModel):
    use_latent_guidance: bool = Field(
        False,
        description="Whether use the latent of the givent image to guide the generation.",
    )
    reference_fidelity: float = Field(
        0.0, description="Fidelity of the reference image."
    )


class ControlNetModel(
    ReturnArraysModel,
    _InpaintingMixin,
    DiffusionModel,
    MaxWHModel,
    _ControlNetModel,
    ImageModel,
):
    pass
