import time

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from fastapi import Response
from pydantic import Field
from cftool.misc import shallow_copy_dict
from cfcv.misc.toolkit import np_to_bytes
from cflearn.api.cv.models.diffusion import ControlNetHints

from .utils import to_canvas
from .common import cleanup
from .common import get_controlnet
from .common import need_change_device
from .common import IAlgorithm
from .common import ControlNetModel
from .control import apply_control
from .control import MLSDModel
from .control import PoseModel
from .control import CannyModel
from .control import DepthModel
from .control import ControlMLSDModel
from .control import ControlPoseModel
from .control import ControlCannyModel
from .control import ControlDepthModel


control_multi_endpoint = "/control/multi"
model_mapping = {
    ControlNetHints.MLSD: ControlMLSDModel,
    ControlNetHints.POSE: ControlPoseModel,
    ControlNetHints.CANNY: ControlCannyModel,
    ControlNetHints.DEPTH: ControlDepthModel,
}


class ControlMultiModel(ControlNetModel):
    types: List[ControlNetHints] = Field(..., description="Target control types")
    depth_params: DepthModel = Field(
        DepthModel(),
        description="Params for depth control.",
    )
    canny_params: CannyModel = Field(
        CannyModel(),
        description="Params for canny control.",
    )
    pose_params: PoseModel = Field(
        PoseModel(),
        description="Params for pose control.",
    )
    mlsd_params: MLSDModel = Field(
        MLSDModel(),
        description="Params for mlsd control.",
    )


def gather_all_data(data: ControlMultiModel) -> Dict[ControlNetHints, ControlNetModel]:
    common = data.dict()
    for k in [
        "types",
        "depth_params",
        "canny_params",
        "pose_params",
        "mlsd_params",
    ]:
        common.pop(k)
    all_data = {}
    for hint_type in data.types:
        h_kw = shallow_copy_dict(common)
        h_kw.update(getattr(data, f"{hint_type}_params").dict())
        all_data[hint_type] = model_mapping[hint_type](**h_kw)
    return all_data


@IAlgorithm.auto_register()
class ControlMulti(IAlgorithm):
    model_class = ControlMultiModel

    endpoint = control_multi_endpoint

    def initialize(self) -> None:
        self.api = get_controlnet()

    async def run(self, data: ControlMultiModel, *args: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = np.array(await self.download_image_with_retry(data.url))
        if not data.hint_url:
            hint_image = image
        else:
            hint_image = np.array(await self.download_image_with_retry(data.hint_url))
        t1 = time.time()
        results, latencies = apply_control(
            gather_all_data(data),
            self.api,
            image,
            hint_image,
            data.types,
        )
        t2 = time.time()
        content = np_to_bytes(to_canvas(results))
        t3 = time.time()
        latencies["download"] = t1 - t0
        latencies["to_canvas"] = t3 - t2
        self.log_times(latencies)
        return Response(content=content, media_type="image/png")


__all__ = [
    "control_multi_endpoint",
    "ControlMultiModel",
    "ControlMulti",
]
