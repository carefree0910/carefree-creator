import cv2
import math
import time
import torch
import asyncio

import numpy as np
import torchvision.transforms as T

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
from transformers import AutoModel
from transformers import AutoFeatureExtractor
from PIL.ImageFilter import GaussianBlur
from cftool.cv import to_rgb
from cftool.cv import to_uint8
from cftool.cv import ImageBox
from cftool.cv import ImageProcessor
from cftool.geometry import Point
from cftool.geometry import Matrix2D
from cftool.geometry import PivotType
from cftool.geometry import ExpandType
from cftool.geometry import Matrix2DProperties
from cfclient.models import ImageModel
from cflearn.misc.toolkit import eval_context

from .utils import get_contrast_bg
from .common import get_response
from .common import IAlgorithm
from .common import ReturnArraysModel


cv_blur_endpoint = "/cv/blur"
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
cv_image_similarity_endpoint = "/cv/similarity"
cv_repositioning_endpoint = "/cv/repositioning"


class CVImageModel(ReturnArraysModel, ImageModel):
    pass


class BlurModel(CVImageModel):
    radius: int = Field(2, description="size of the kernel")


@IAlgorithm.auto_register()
class Blur(IAlgorithm):
    model_class = BlurModel

    endpoint = cv_blur_endpoint

    def initialize(self) -> None:
        pass

    async def run(self, data: BlurModel, *args: Any, **kwargs: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        array = np.array(image.filter(GaussianBlur(data.radius)))
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


@IAlgorithm.auto_register()
class Grayscale(IAlgorithm):
    model_class = CVImageModel

    endpoint = cv_grayscale_endpoint

    def initialize(self) -> None:
        pass

    async def run(self, data: CVImageModel, *args: Any, **kwargs: Any) -> Response:
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


class ErodeModel(CVImageModel):
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


class ResizeModel(ResamplingModel, CVImageModel):
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


class GetMaskModel(CVImageModel):
    get_inverse: bool = Field(False, description="Whether get the inverse mask.")
    binarize_threshold: Optional[int] = Field(
        None,
        ge=0,
        le=255,
        description="If not None, will binarize the mask with this value.",
    )


@IAlgorithm.auto_register()
class GetMask(IAlgorithm):
    model_class = GetMaskModel

    endpoint = cv_get_mask_endpoint

    def initialize(self) -> None:
        pass

    async def run(self, data: GetMaskModel, *args: Any, **kwargs: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        if image.mode == "RGBA":
            mask = np.array(image)[..., -1]
        else:
            mask = np.array(image.convert("L"))
        if data.get_inverse:
            mask = 255 - mask
        if data.binarize_threshold is not None:
            mask = np.where(mask > data.binarize_threshold, 255, 0)
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


class HistogramMatchModel(CVImageModel):
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


class ImageSimilarityModel(ReturnArraysModel):
    url_0: Union[str, List[str]]
    url_1: Union[str, List[str]]
    batch_size: int = Field(4, description="batch size")
    skip_to_rgb: bool = False


class ImageSimilarityResponse(BaseModel):
    similarity: Union[float, List[List[float]]]


@IAlgorithm.auto_register()
class ImageSimilarity(IAlgorithm):
    model_class = ImageSimilarityModel
    response_model_class = ImageSimilarityResponse

    endpoint = cv_image_similarity_endpoint

    def initialize(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_ckpt = "nateraw/vit-base-beans"
        extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt).to(device)
        self.transform = T.Compose(
            [
                T.Resize(int((256 / 224) * extractor.size["height"])),
                T.CenterCrop(extractor.size["height"]),
                T.ToTensor(),
                T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
            ]
        )

    async def run(
        self,
        data: ImageSimilarityModel,
        *args: Any,
        **kwargs: Any,
    ) -> ImageSimilarityResponse:
        self.log_endpoint(data)
        t0 = time.time()
        if isinstance(data.url_0, str) and isinstance(data.url_1, str):
            is_item = True
            futures = [
                self.get_image_from("url_0", data, kwargs),
                self.get_image_from("url_1", data, kwargs),
            ]
            im0_indices = [0]
            im1_indices = [1]
        else:
            is_item = False
            url_0 = data.url_0 if isinstance(data.url_0, list) else [data.url_0]
            url_1 = data.url_1 if isinstance(data.url_1, list) else [data.url_1]
            all_urls = {}
            im0_indices = []
            im1_indices = []
            for url in url_0:
                im0_indices.append(all_urls.setdefault(url, len(all_urls)))
            for url in url_1:
                im1_indices.append(all_urls.setdefault(url, len(all_urls)))
            reverse_urls = {v: k for k, v in all_urls.items()}
            sorted_urls = [reverse_urls[i] for i in range(len(all_urls))]
            futures = list(map(self.download_image_with_retry, sorted_urls))
        images = await asyncio.gather(*futures)
        t1 = time.time()
        if not data.skip_to_rgb:
            images = list(map(to_rgb, images))
        t2 = time.time()
        embeddings = self._extract_embeddings(images, data.batch_size)
        t3 = time.time()
        e0 = embeddings[im0_indices]
        e1 = embeddings[im1_indices]
        sim = (e0 @ e1.t()) / (e0.norm(dim=-1)[..., None] * e1.norm(dim=-1)[None])
        if is_item:
            sim = sim.item()
        else:
            sim = sim.cpu().numpy().tolist()
        self.log_times(
            {
                "download": t1 - t0,
                "to_rgb": t2 - t1,
                "inference": t3 - t2,
                "calculation": time.time() - t3,
            }
        )
        return ImageSimilarityResponse(similarity=sim)

    def _extract_embeddings(
        self,
        images: List[Image.Image],
        batch_size: int,
    ) -> torch.Tensor:
        transformed = torch.stack(list(map(self.transform, images)))
        with eval_context(self.model):
            num_batches = (len(images) - 1) // batch_size + 1
            embeddings = []
            for i in range(num_batches):
                i_transformed = transformed[i * batch_size : (i + 1) * batch_size]
                batch = {"pixel_values": i_transformed.to(self.model.device)}
                embeddings.append(self.model(**batch).last_hidden_state[:, 0].cpu())
        return torch.cat(embeddings)


class FillType(str, Enum):
    FIT = "fit"
    IOU = "iou"
    COVER = "cover"


class AlignType(str, Enum):
    CENTER = "center"
    CENTROID = "centroid"
    HALF_CENTROID = "half-centroid"


class AffineFrameModel(BaseModel):
    x: float
    y: float
    w: float
    h: float
    rotation: float
    frame_w: float
    frame_h: float

    @property
    def wh_ratio(self) -> float:
        return self.w / self.h

    @property
    def frame_wh_ratio(self) -> float:
        return self.frame_w / self.frame_h

    def scale_to(self, ow: float, oh: float) -> "Matrix2D":
        return Matrix2D.from_properties(
            Matrix2DProperties(
                x=self.x,
                y=self.y,
                w=self.w,
                h=self.h,
                theta=self.rotation * math.pi / 180,
                skew_x=0,
                skew_y=0,
            )
        ).scale(ow / self.frame_w, oh / self.frame_h, center=Point.origin())


class ScaleByDensityModel(BaseModel):
    min_scale: float = Field(
        0.8,
        description="The minimum scale used when density is large.",
    )
    max_scale: float = Field(
        1.1,
        description="The maximum scale used when density is small.",
    )


class RepositioningModel(ResamplingModel, CVImageModel):
    url: str = Field(..., description="The `cdn` / `cos` url of the goods.")
    output_width: int = Field(800, description="The width of the output image.")
    output_height: int = Field(800, description="The height of the output image.")
    binary_threshold: int = Field(127, description="The binary threshold.")
    stick_to_edge: bool = Field(
        True,
        description="Whether enable 'stick to edge' logic.",
    )
    stick_distance_threshold: int = Field(
        3,
        description="The distance threshold of 'sticking'.",
    )
    stick_image_fill_type: FillType = Field(
        FillType.FIT,
        description="The fill type under 'stick to edge' logic.",
    )
    stick_priorities: str = Field("tlbr", description="The priorities of 'sticking'.")
    affine_frame_list: Optional[List[AffineFrameModel]] = Field(
        None,
        description="The target affine frames to match.",
    )
    fill_type: FillType = Field(FillType.FIT, description="The fill type.")
    align_type: AlignType = Field(AlignType.CENTER, description="The align type.")
    scale_by_density: bool = Field(
        True,
        description="Whether enable 'scale by density' logic.",
    )
    scale_by_density_params: Optional[ScaleByDensityModel] = Field(
        None,
        description="The params of 'scale by density' logic.",
    )
    wh_limit: int = Field(
        16384,
        description="maximum width or height of the output image",
    )


opposite_edges = dict(l="r", t="b", r="l", b="t")


@IAlgorithm.auto_register()
class Repositioning(IAlgorithm):
    model_class = RepositioningModel

    endpoint = cv_repositioning_endpoint

    def initialize(self) -> None:
        pass

    async def run(
        self,
        data: RepositioningModel,
        *args: Any,
        **kwargs: Any,
    ) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        image = await self.get_image_from("url", data, kwargs)
        t1 = time.time()
        ow, oh = data.output_width, data.output_height
        image = image.convert("RGBA")
        array = np.array(image)
        alpha = array[..., -1]
        image_box = ImageBox.from_mask(alpha, data.binary_threshold)
        img_w, img_h = image.size
        sticky_edges = set()
        if data.stick_to_edge:
            if image_box.l < data.stick_distance_threshold:
                sticky_edges.add("l")
            if image_box.t < data.stick_distance_threshold:
                sticky_edges.add("t")
            if image_box.r > img_w - data.stick_distance_threshold:
                sticky_edges.add("r")
            if image_box.b > img_h - data.stick_distance_threshold:
                sticky_edges.add("b")
        t2 = time.time()
        mask = None
        mask_sum = None
        if sticky_edges:
            fill_type = data.stick_image_fill_type
            image_wh_ratio = img_w / img_h
            frame_matrix = Matrix2D(a=ow, b=0, c=0, d=oh, e=0, f=0)
            density_scale = None
        else:
            image = image.crop(image_box.tuple)
            img_w, img_h = image.size
            alpha = alpha[image_box.t : image_box.b, image_box.l : image_box.r]
            if not data.scale_by_density:
                density_scale = None
            else:
                density_scale_params = data.scale_by_density_params
                if density_scale_params is None:
                    density_scale_params = ScaleByDensityModel()
                min_density_scale = density_scale_params.min_scale
                max_density_scale = density_scale_params.max_scale
                mask = alpha > data.binary_threshold
                mask_sum = mask.sum()
                density = 1.0 - mask_sum / (max(image_box.w, image_box.h) ** 2)
                density_scale = (
                    min_density_scale
                    + (max_density_scale - min_density_scale) * density
                )
            fill_type = data.fill_type
            affine_frames = data.affine_frame_list
            if affine_frames is None:
                d = dict(x=40, y=40, w=720, h=720, rotation=0, frame_w=800, frame_h=800)
                affine_frames = [AffineFrameModel(**d)]
            canvas_wh_ratio = ow / oh
            filtered = [
                frame
                for frame in affine_frames
                if abs(frame.frame_wh_ratio - canvas_wh_ratio) < 1.0e-6
            ]
            if not filtered:
                raise ValueError("no affine frame matches the output ratio")
            wh_ratios = [frame.wh_ratio for frame in filtered]
            image_wh_ratio = image_box.wh_ratio
            best_index = np.argmin([abs(ratio - image_wh_ratio) for ratio in wh_ratios])
            frame_matrix = filtered[best_index].scale_to(ow, oh)
        frame_wh_ratio = frame_matrix.abs_wh_ratio
        if fill_type == FillType.IOU:
            expand_type = ExpandType.IOU
        elif fill_type == FillType.FIT:
            expand_type = (
                ExpandType.FIX_H
                if frame_wh_ratio > image_wh_ratio
                else ExpandType.FIX_W
            )
        else:
            expand_type = (
                ExpandType.FIX_H
                if frame_wh_ratio < image_wh_ratio
                else ExpandType.FIX_W
            )
        expanded_image_matrix = frame_matrix.set_wh_ratio(
            image_wh_ratio,
            type=expand_type,
            pivot=PivotType.CENTER,
        )
        if density_scale is not None:
            expanded_image_matrix = expanded_image_matrix.scale(
                density_scale,
                density_scale,
                center=expanded_image_matrix.pivot(PivotType.CENTER),
            )
        expanded_properties = expanded_image_matrix.decompose()
        ew, eh = expanded_properties.w, expanded_properties.h
        ew_ratio = ew / img_w
        eh_ratio = eh / img_h
        expanded_properties.w = ew_ratio
        expanded_properties.h = eh_ratio
        delta = None
        if sticky_edges:
            sticked = set()
            for edge in data.stick_priorities:
                if edge not in sticky_edges:
                    continue
                if opposite_edges[edge] in sticked:
                    continue
                if edge == "l":
                    expanded_properties.x = 0
                elif edge == "t":
                    expanded_properties.y = 0
                elif edge == "r":
                    expanded_properties.x = ow - ew
                elif edge == "b":
                    expanded_properties.y = oh - eh
                sticked.add(edge)
        elif data.align_type != AlignType.CENTER:
            if mask is None:
                mask = alpha > data.binary_threshold
            if mask_sum is None:
                mask_sum = mask.sum()
            x_center = 0.5 * img_w
            y_center = 0.5 * img_h
            x_centroid = (mask * np.arange(img_w)[None]).sum() / mask_sum
            y_centroid = (mask * np.arange(img_h)[..., None]).sum() / mask_sum
            x_delta = x_center - x_centroid.item()
            y_delta = y_center - y_centroid.item()
            if data.align_type == AlignType.HALF_CENTROID:
                x_delta *= 0.5
                y_delta *= 0.5
            delta = Point(x_delta, y_delta).rotate(frame_matrix.theta)
        final_matrix = Matrix2D.from_properties(expanded_properties)
        if delta is not None:
            final_matrix = final_matrix.move(delta)
        affined = affine(
            image,
            final_matrix.a,
            final_matrix.b,
            final_matrix.c,
            final_matrix.d,
            final_matrix.e,
            final_matrix.f,
            ow,
            oh,
            data.resampling,
            data.wh_limit,
        )
        t3 = time.time()
        res = get_response(data, [affined])
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
    "cv_blur_endpoint",
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
    "cv_image_similarity_endpoint",
    "cv_repositioning_endpoint",
    "CVImageModel",
    "BlurModel",
    "ErodeModel",
    "ResizeModel",
    "BaseAffineModel",
    "AffineModel",
    "FillBGModel",
    "LTRBModel",
    "ModifyBoxModel",
    "GenerateMasksModel",
    "CropImageModel",
    "HistogramMatchModel",
    "ImageSimilarityModel",
    "ImageSimilarityResponse",
    "RepositioningModel",
    "Blur",
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
    "ImageSimilarity",
    "Repositioning",
]
