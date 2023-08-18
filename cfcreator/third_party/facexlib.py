import os
import cv2
import time
import torch

import numpy as np

from enum import Enum
from typing import Any
from typing import List
from fastapi import Response
from pydantic import Field
from pydantic import BaseModel
from cflearn.misc.toolkit import eval_context
from torchvision.transforms.functional import normalize

from .utils import external_folder
from ..cv import CVImageModel
from ..common import get_response
from ..common import IAlgorithm

try:
    from facexlib.parsing import init_parsing_model
    from facexlib.detection import init_detection_model
except:
    init_parsing_model = None
    init_detection_model = None


ROOT = os.path.join(external_folder, "facexlib")


facexlib_parse_endpoint = "/third_party/facexlib/parse"
facexlib_detect_endpoint = "/third_party/facexlib/detect"


class FaceAreas(str, Enum):
    FACE = "face"
    NECK = "neck"
    HAIR = "hair"
    HAT = "hat"


class FacexlibParseModel(CVImageModel):
    mask_size: int = Field(0, description="mask size")
    affected_areas: List[FaceAreas] = Field(
        default_factory=lambda: [FaceAreas.FACE],
        description="affected face areas",
    )


@IAlgorithm.auto_register()
class FacexlibParse(IAlgorithm):
    model_class = FacexlibParseModel

    endpoint = facexlib_parse_endpoint

    def initialize(self) -> None:
        if init_parsing_model is None:
            raise ImportError("`facexlib` should be installed for `FacexlibParse`")
        self.model = init_parsing_model(device="cpu", model_rootpath=ROOT)

    async def run(
        self, data: FacexlibParseModel, *args: Any, **kwargs: Any
    ) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        model = self.model.to("cuda")
        t2 = time.time()
        w, h = image.size
        array = np.array(image)
        if w != 512 or h != 512:
            rw = (int(w * (512 / w)) // 8) * 8
            rh = (int(h * (512 / h)) // 8) * 8
            array = cv2.resize(array, dsize=(rw, rh))
        array = array.astype(np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1))
        normalize(tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        tensor = torch.unsqueeze(tensor, 0).to("cuda")
        with eval_context(model):
            face = model(tensor)[0]
        t3 = time.time()
        face = face.squeeze(0).cpu().numpy().argmax(0)
        face = face.copy().astype(np.uint8)
        mask = self._to_mask(face, data.affected_areas)
        if data.mask_size > 0:
            mask = cv2.dilate(
                mask,
                np.ones((5, 5), np.uint8),
                iterations=data.mask_size,
            )
        if w != 512 or h != 512:
            mask = cv2.resize(mask, dsize=(w, h))
        t4 = time.time()
        self.model.to("cpu")
        t5 = time.time()
        res = get_response(data, [mask])
        self.log_times(
            {
                "download": t1 - t0,
                "to_cuda": t2 - t1,
                "inference": t3 - t2,
                "postprocess": t4 - t3,
                "cleanup": t5 - t4,
                "get_response": time.time() - t5,
            }
        )
        return res

    def _to_mask(self, face: np.ndarray, affected_areas: List[FaceAreas]) -> np.ndarray:
        area_set = set(affected_areas)
        keep_face = FaceAreas.FACE in area_set
        keep_neck = FaceAreas.NECK in area_set
        keep_hair = FaceAreas.HAIR in area_set
        keep_hat = FaceAreas.HAT in area_set

        mask = np.zeros((face.shape[0], face.shape[1], 3), dtype=np.uint8)
        num_of_class = np.max(face)
        for i in range(1, num_of_class + 1):
            index = np.where(face == i)
            if i < 14 and keep_face:
                mask[index[0], index[1], :] = [255, 255, 255]
            elif i == 14 and keep_neck:
                mask[index[0], index[1], :] = [255, 255, 255]
            elif i == 17 and keep_hair:
                mask[index[0], index[1], :] = [255, 255, 255]
            elif i == 18 and keep_hat:
                mask[index[0], index[1], :] = [255, 255, 255]
        return mask


class FacexlibDetectModel(CVImageModel):
    confidence: float = Field(0.97, description="confidence threshold")


class FacexlibDetectResponse(BaseModel):
    ltrbs: List[List[int]]


@IAlgorithm.auto_register()
class FacexlibDetect(IAlgorithm):
    model_class = FacexlibDetectModel

    endpoint = facexlib_detect_endpoint

    def initialize(self) -> None:
        if init_detection_model is None:
            raise ImportError("`facexlib` should be installed for `FacexlibDetect`")
        self.model = init_detection_model(
            "retinaface_resnet50",
            device="cuda",
            model_rootpath=ROOT,
        )
        self.model.to("cpu")

    async def run(
        self, data: FacexlibDetectModel, *args: Any, **kwargs: Any
    ) -> FacexlibDetectResponse:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        model = self.model.to("cuda")
        t2 = time.time()
        with eval_context(model):
            boxes, _ = model.align_multi(image, data.confidence)
        t3 = time.time()
        ltrbs = [list(map(int, box[:4])) for box in boxes]
        t4 = time.time()
        self.model.to("cpu")
        self.log_times(
            {
                "download": t1 - t0,
                "to_cuda": t2 - t1,
                "inference": t3 - t2,
                "postprocess": t4 - t3,
                "cleanup": time.time() - t4,
            }
        )
        return FacexlibDetectResponse(ltrbs=ltrbs)


__all__ = [
    "facexlib_parse_endpoint",
    "facexlib_detect_endpoint",
    "FaceAreas",
    "FacexlibParseModel",
    "FacexlibDetectModel",
    "FacexlibDetectResponse",
    "FacexlibParse",
    "FacexlibDetect",
]
