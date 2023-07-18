import cv2
import time

import numpy as np

from PIL import Image
from enum import Enum
from typing import Any
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from fastapi import Response
from pydantic import Field
from pydantic import BaseModel
from cftool.cv import to_rgb
from cftool.cv import to_uint8
from cftool.cv import np_to_bytes
from cftool.cv import ImageBox
from cftool.cv import ImageProcessor
from cftool.geometry import Matrix2D
from cfclient.models import ImageModel

from .utils import get_contrast_bg
from .common import get_response
from .common import IAlgorithm
from .common import ReturnArraysModel


cv_grayscale_endpoint = "/cv/grayscale"
cv_erode_endpoint = "/cv/erode"
cv_resize_endpoint = "/cv/resize"
cv_affine_endpoint = "/cv/affine"
cv_get_mask_endpoint = "/cv/get_mask"
cv_inverse_endpoint = "/cv/inverse"
cv_fill_bg_endpoint = "/cv/fill_bg"
cv_get_size_endpoint = "/cv/get_size"
cv_modify_box_endpoint = "/cv/modify_box"
cv_generate_masks_endpoint = "/cv/generate_masks"
cv_crop_image_endpoint = "/cv/crop_image"
cv_histogram_match_endpoint = "/cv/hist_match"


class GrayscaleModel(ReturnArraysModel, ImageModel):
    pass


@IAlgorithm.auto_register()
class Grayscale(IAlgorithm):
    model_class = GrayscaleModel

    endpoint = cv_grayscale_endpoint

    def initialize(self) -> None:
        pass

    async def run(self, data: GrayscaleModel, *args: Any, **kwargs: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        array = np.array(image.convert("L"))
        t2 = time.time()
        res = get_response(data, [array])
        self.log_times(
            {
                "download": t1 - t0,
                "process": t2 - t1,
                "get_response": time.time() - t2,
            }
        )
        return res


def erode(array: np.ndarray, n_iter: int, kernel_size: int) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(array, kernel, iterations=n_iter)


class Resampling(str, Enum):
    NEAREST = "nearest"
    BOX = "box"
    BILINEAR = "bilinear"
    HAMMING = "hamming"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"


class ResamplingModel(BaseModel):
    resampling: Resampling = Field(Resampling.BILINEAR, description="resampling method")


def resize(image: Image.Image, w: int, h: int, resampling: Resampling) -> Image.Image:
    if resampling == Resampling.NEAREST:
        r = Image.NEAREST
    elif resampling == Resampling.BOX:
        r = Image.BOX
    elif resampling == Resampling.BILINEAR:
        r = Image.BILINEAR
    elif resampling == Resampling.HAMMING:
        r = Image.HAMMING
    elif resampling == Resampling.BICUBIC:
        r = Image.BICUBIC
    else:
        r = Image.LANCZOS
    return image.resize((w, h), r)


def affine(
    image: Image.Image,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    w: int,
    h: int,
    resampling: Resampling,
    wh_limit: int,
) -> np.ndarray:
    matrix2d = Matrix2D(a=a, b=b, c=c, d=d, e=e, f=f)
    properties = matrix2d.decompose()
    iw, ih = image.size
    nw = max(round(iw * abs(properties.w)), 1)
    nh = max(round(ih * abs(properties.h)), 1)
    if nw > wh_limit or nh > wh_limit:
        raise ValueError(f"image size ({nw}, {nh}) exceeds wh_limit ({wh_limit})")
    array = np.array(resize(image, nw, nh, resampling))
    properties.w = 1
    properties.h = 1 if properties.h > 0 else -1
    matrix2d = Matrix2D.from_properties(properties)
    return cv2.warpAffine(array, matrix2d.matrix, [w, h])


class ErodeModel(ReturnArraysModel, ImageModel):
    n_iter: int = Field(1, description="number of iterations")
    kernel_size: int = Field(3, description="size of the kernel")
    threshold: int = Field(0, description="threshold of the alpha channel")
    padding: int = Field(8, description="padding for the image")


@IAlgorithm.auto_register()
class Erode(IAlgorithm):
    model_class = ErodeModel

    endpoint = cv_erode_endpoint

    def initialize(self) -> None:
        pass

    async def run(self, data: ErodeModel, *args: Any, **kwargs: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        w, h = image.size
        array = np.array(image.convert("RGBA"))
        alpha = array[..., -1]
        padded = np.pad(alpha, (data.padding, data.padding), constant_values=0)
        binarized = cv2.threshold(padded, data.threshold, 255, cv2.THRESH_BINARY)[1]
        eroded = erode(binarized, data.n_iter, data.kernel_size)
        shrinked = eroded[data.padding : -data.padding, data.padding : -data.padding]
        merged_alpha = np.minimum(alpha, shrinked)
        array[..., -1] = merged_alpha
        rgb = array[..., :3].reshape([-1, 3])
        rgb[(merged_alpha == 0).ravel()] = 0
        array[..., :3] = rgb.reshape([h, w, 3])
        t2 = time.time()
        res = get_response(data, [array])
        self.log_times(
            {
                "download": t1 - t0,
                "process": t2 - t1,
                "get_response": time.time() - t2,
            }
        )
        return res


class ResizeModel(ReturnArraysModel, ResamplingModel, ImageModel):
    w: int = Field(..., description="width of the output image")
    h: int = Field(..., description="width of the output image")


@IAlgorithm.auto_register()
class Resize(IAlgorithm):
    model_class = ResizeModel

    endpoint = cv_resize_endpoint

    def initialize(self) -> None:
        pass

    async def run(self, data: ResizeModel, *args: Any, **kwargs: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        resized = np.array(resize(image, data.w, data.h, data.resampling))
        t2 = time.time()
        res = get_response(data, [resized])
        self.log_times(
            {
                "download": t1 - t0,
                "process": t2 - t1,
                "get_response": time.time() - t2,
            }
        )
        return res


class BaseAffineModel(BaseModel):
    a: float = Field(..., description="`a` of the affine matrix")
    b: float = Field(..., description="`b` of the affine matrix")
    c: float = Field(..., description="`c` of the affine matrix")
    d: float = Field(..., description="`d` of the affine matrix")
    e: float = Field(..., description="`e` of the affine matrix")
    f: float = Field(..., description="`f` of the affine matrix")
    wh_limit: int = Field(
        16384,
        description="maximum width or height of the output image",
    )


class AffineModel(ResizeModel, BaseAffineModel):
    pass


@IAlgorithm.auto_register()
class Affine(IAlgorithm):
    model_class = AffineModel

    endpoint = cv_affine_endpoint

    def initialize(self) -> None:
        pass

    async def run(self, data: AffineModel, *args: Any, **kwargs: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        output = affine(
            image,
            data.a,
            data.b,
            data.c,
            data.d,
            data.e,
            data.f,
            data.w,
            data.h,
            data.resampling,
            data.wh_limit,
        )
        t2 = time.time()
        res = get_response(data, [output])
        self.log_times(
            {
                "download": t1 - t0,
                "process": t2 - t1,
                "get_response": time.time() - t2,
            }
        )
        return res


class CVImageModel(ReturnArraysModel, ImageModel):
    pass


@IAlgorithm.auto_register()
class GetMask(IAlgorithm):
    model_class = CVImageModel

    endpoint = cv_get_mask_endpoint

    def initialize(self) -> None:
        pass

    async def run(self, data: CVImageModel, *args: Any, **kwargs: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        if image.mode == "RGBA":
            mask = np.array(image)[..., -1]
        else:
            mask = np.array(image.convert("L"))
        t2 = time.time()
        res = get_response(data, [mask])
        self.log_times(
            {
                "download": t1 - t0,
                "process": t2 - t1,
                "get_response": time.time() - t2,
            }
        )
        return res


@IAlgorithm.auto_register()
class Inverse(IAlgorithm):
    model_class = CVImageModel

    endpoint = cv_inverse_endpoint

    def initialize(self) -> None:
        pass

    async def run(self, data: CVImageModel, *args: Any, **kwargs: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        array = np.array(image)
        inversed = 255 - array
        t2 = time.time()
        res = get_response(data, [inversed])
        self.log_times(
            {
                "download": t1 - t0,
                "process": t2 - t1,
                "get_response": time.time() - t2,
            }
        )
        return res


class FillBGModel(CVImageModel):
    bg: Optional[Union[int, Tuple[int, int, int]]] = Field(
        None,
        description="""
Target background color.
> If not specified, `get_contrast_bg` will be used to calculate the `bg`.
""",
    )


@IAlgorithm.auto_register()
class FillBG(IAlgorithm):
    model_class = FillBGModel

    endpoint = cv_fill_bg_endpoint

    def initialize(self) -> None:
        pass

    async def run(self, data: FillBGModel, *args: Any, **kwargs: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        bg = data.bg
        if bg is None:
            bg = get_contrast_bg(image)
        if isinstance(bg, int):
            bg = bg, bg, bg
        image = to_rgb(image, bg)
        t2 = time.time()
        res = get_response(data, [np.array(image)])
        self.log_times(
            {
                "download": t1 - t0,
                "process": t2 - t1,
                "get_response": time.time() - t2,
            }
        )
        return res


@IAlgorithm.auto_register()
class GetSize(IAlgorithm):
    model_class = ImageModel

    endpoint = cv_get_size_endpoint

    def initialize(self) -> None:
        pass

    async def run(self, data: ImageModel, *args: Any, **kwargs: Any) -> List[int]:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        w, h = image.size
        self.log_times(
            {
                "download": t1 - t0,
                "process": time.time() - t1,
            }
        )
        return [w, h]


class LTRBModel(BaseModel):
    lt_rb: Tuple[int, int, int, int] = Field(
        ...,
        description="The left-top and right-bottom points.",
    )


class ModifyBoxModel(LTRBModel):
    w: Optional[int] = Field(None, description="The width of the image.")
    h: Optional[int] = Field(None, description="The height of the image.")
    padding: int = Field(0, description="The padding size.")
    to_square: bool = Field(False, description="Turn the box into a square box.")


@IAlgorithm.auto_register()
class ModifyBox(IAlgorithm):
    model_class = ModifyBoxModel

    endpoint = cv_modify_box_endpoint

    def initialize(self) -> None:
        pass

    async def run(
        self, data: ModifyBoxModel, *args: Any, **kwargs: Any
    ) -> List[List[int]]:
        self.log_endpoint(data)
        t0 = time.time()
        w = data.w
        h = data.h
        padding = data.padding
        box = ImageBox(*data.lt_rb)
        box = box.pad(padding, w=w, h=h)
        if data.to_square:
            box = box.to_square()
        self.log_times({"process": time.time() - t0})
        return [list(box.tuple)]


class GenerateMasksModel(ReturnArraysModel):
    w: int = Field(..., description="The width of the canvas.")
    h: int = Field(..., description="The height of the canvas.")
    lt_rb_list: List[Tuple[int, int, int, int]] = Field(..., description="The boxes.")
    merge: bool = Field(False, description="Whether merge the masks.")


@IAlgorithm.auto_register()
class GenerateMasks(IAlgorithm):
    model_class = GenerateMasksModel

    endpoint = cv_generate_masks_endpoint

    def initialize(self) -> None:
        pass

    async def run(
        self, data: GenerateMasksModel, *args: Any, **kwargs: Any
    ) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        canvas = np.zeros((data.h, data.w), dtype=np.uint8)
        results = []
        for l, t, r, b in data.lt_rb_list:
            i_canvas = canvas if data.merge else canvas.copy()
            i_canvas[t:b, l:r] = 255
            if not data.merge:
                results.append(i_canvas)
        if data.merge:
            results.append(canvas)
        t1 = time.time()
        res = get_response(data, results)
        self.log_times(
            {
                "process": t1 - t0,
                "get_response": time.time() - t1,
            }
        )
        return res


class CropImageModel(CVImageModel, LTRBModel):
    pass


@IAlgorithm.auto_register()
class CropImage(IAlgorithm):
    model_class = CropImageModel

    endpoint = cv_crop_image_endpoint

    def initialize(self) -> None:
        pass

    async def run(self, data: CropImageModel, *args: Any, **kwargs: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        l, t, r, b = data.lt_rb
        image = image.crop((l, t, r, b))
        t2 = time.time()
        res = get_response(data, [np.array(image)])
        self.log_times(
            {
                "download": t1 - t0,
                "process": t2 - t1,
                "get_response": time.time() - t2,
            }
        )
        return res


class HistogramMatchModel(ReturnArraysModel, ImageModel):
    bg_url: str = Field(..., description="The `cdn` / `cos` url of the background.")
    use_hsv: bool = Field(False, description="Whether use the HSV space to match.")
    strength: float = Field(1.0, description="Strength of the matching.")


@IAlgorithm.auto_register()
class HistogramMatch(IAlgorithm):
    model_class = HistogramMatchModel

    endpoint = cv_histogram_match_endpoint

    def initialize(self) -> None:
        pass

    async def run(
        self,
        data: HistogramMatchModel,
        *args: Any,
        **kwargs: Any,
    ) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        bg = await self.get_image_from("bg_url", data, kwargs)
        t1 = time.time()
        to_normalized = lambda im: np.array(im).astype(np.float32) / 255.0
        rgba, bg_arr = map(to_normalized, [image, bg])
        rgb, alpha = rgba[..., :3], rgba[..., -1:]
        merged = rgb * alpha + bg_arr * (1.0 - alpha)
        merged, bg_arr = map(to_uint8, [merged, bg_arr])
        t2 = time.time()
        if data.use_hsv:
            merged = cv2.cvtColor(merged, cv2.COLOR_RGB2HSV)
            bg_arr = cv2.cvtColor(bg_arr, cv2.COLOR_RGB2HSV)
        adjusted = ImageProcessor.match_histograms(
            merged,
            bg_arr,
            alpha[..., 0] > 0,
            strength=data.strength,
        )
        if data.use_hsv:
            adjusted = cv2.cvtColor(adjusted, cv2.COLOR_HSV2RGB)
        t3 = time.time()
        res = get_response(data, [adjusted])
        self.log_times(
            {
                "download": t1 - t0,
                "preprocess": t2 - t1,
                "calculation": t3 - t2,
                "get_response": time.time() - t3,
            }
        )
        return res


__all__ = [
    "cv_grayscale_endpoint",
    "cv_erode_endpoint",
    "cv_resize_endpoint",
    "cv_affine_endpoint",
    "cv_get_mask_endpoint",
    "cv_inverse_endpoint",
    "cv_fill_bg_endpoint",
    "cv_get_size_endpoint",
    "cv_modify_box_endpoint",
    "cv_generate_masks_endpoint",
    "cv_crop_image_endpoint",
    "cv_histogram_match_endpoint",
    "GrayscaleModel",
    "ErodeModel",
    "ResizeModel",
    "BaseAffineModel",
    "AffineModel",
    "CVImageModel",
    "FillBGModel",
    "LTRBModel",
    "ModifyBoxModel",
    "GenerateMasksModel",
    "CropImageModel",
    "HistogramMatchModel",
    "Grayscale",
    "Erode",
    "Resize",
    "Affine",
    "GetMask",
    "Inverse",
    "FillBG",
    "GetSize",
    "ModifyBox",
    "GenerateMasks",
    "CropImage",
    "HistogramMatch",
]
