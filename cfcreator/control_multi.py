import time

from typing import Any
from typing import List
from typing import Union
from fastapi import Response
from pydantic import Field
from pydantic import BaseModel
from cftool.cv import np_to_bytes

from .utils import to_canvas
from .common import register_sd
from .common import register_sd_inpainting
from .common import IAlgorithm
from .common import ControlNetModel
from .control import apply_control
from .control import ControlNetHints
from .control import ControlMLSDModel
from .control import ControlPoseModel
from .control import ControlCannyModel
from .control import ControlDepthModel


new_control_multi_endpoint = "/control_new/multi"


class MLSDBundle(BaseModel):
    type: ControlNetHints = Field(ControlNetHints.MLSD, const=True)
    data: ControlMLSDModel


class PoseBundle(BaseModel):
    type: ControlNetHints = Field(ControlNetHints.POSE, const=True)
    data: ControlPoseModel


class CannyBundle(BaseModel):
    type: ControlNetHints = Field(ControlNetHints.CANNY, const=True)
    data: ControlCannyModel


class DepthBundle(BaseModel):
    type: ControlNetHints = Field(ControlNetHints.DEPTH, const=True)
    data: ControlDepthModel


TBundle = Union[MLSDBundle, PoseBundle, CannyBundle, DepthBundle]


class ControlMultiModel(ControlNetModel):
    controls: List[TBundle] = Field(..., description="Target controls")


@IAlgorithm.auto_register()
class ControlMulti(IAlgorithm):
    model_class = ControlMultiModel

    endpoint = new_control_multi_endpoint

    def initialize(self) -> None:
        register_sd()
        register_sd_inpainting()

    async def run(self, data: ControlMultiModel, *args: Any) -> Response:
        self.log_endpoint(data)
        results, latencies = await apply_control(self, data, data.controls)
        t0 = time.time()
        content = None if data.return_arrays else np_to_bytes(to_canvas(results))
        t1 = time.time()
        latencies["to_canvas"] = t1 - t0
        self.log_times(latencies)
        if data.return_arrays:
            return results
        return Response(content=content, media_type="image/png")


__all__ = [
    "new_control_multi_endpoint",
    "ControlMultiModel",
    "ControlMulti",
]
