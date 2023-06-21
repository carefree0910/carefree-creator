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
from .control import _MLSDModel
from .control import _PoseModel
from .control import _CannyModel
from .control import _DepthModel
from .control import ControlNetHints
from .control import ControlNetModelPlaceholder


new_control_multi_endpoint = "/control_new/multi"


class MLSDBundleData(_MLSDModel, ControlNetModelPlaceholder):
    pass


class MLSDBundle(BaseModel):
    type: ControlNetHints = Field(ControlNetHints.MLSD, const=True)
    data: MLSDBundleData


class PoseBundleData(_PoseModel, ControlNetModelPlaceholder):
    pass


class PoseBundle(BaseModel):
    type: ControlNetHints = Field(ControlNetHints.POSE, const=True)
    data: PoseBundleData


class CannyBundleData(_CannyModel, ControlNetModelPlaceholder):
    pass


class CannyBundle(BaseModel):
    type: ControlNetHints = Field(ControlNetHints.CANNY, const=True)
    data: CannyBundleData


class DepthBundleData(_DepthModel, ControlNetModelPlaceholder):
    pass


class DepthBundle(BaseModel):
    type: ControlNetHints = Field(ControlNetHints.DEPTH, const=True)
    data: DepthBundleData


class UniversalControlModel(ControlNetModelPlaceholder):
    pass


class UniversalBundle(BaseModel):
    type: str
    data: UniversalControlModel


TBundle = Union[MLSDBundle, PoseBundle, CannyBundle, DepthBundle, UniversalBundle]


class ControlMultiModel(ControlNetModel):
    controls: List[TBundle] = Field(..., description="Target controls")


@IAlgorithm.auto_register()
class ControlMulti(IAlgorithm):
    model_class = ControlMultiModel

    endpoint = new_control_multi_endpoint

    def initialize(self) -> None:
        register_sd()
        register_sd_inpainting()

    async def run(self, data: ControlMultiModel, *args: Any, **kwargs: Any) -> Response:
        self.log_endpoint(data)
        results, latencies = await apply_control(self, data, data.controls, **kwargs)
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
