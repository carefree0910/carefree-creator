# `apis` sdk is used to prgrammatically call the algorithms.

import time

from cfcreator import *
from PIL import Image
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from pydantic import BaseModel
from collections import OrderedDict
from cftool.misc import random_hash
from cftool.misc import shallow_copy_dict
from cfclient.core import HttpClient
from cfclient.utils import download_image_with_retry
from cfclient.models import TextModel
from cfclient.models import ImageModel
from cfclient.models import algorithms as registered_algorithms
from cflearn.api.cv.diffusion import ControlNetHints


TRes = Union[List[str], List[Image.Image]]
UPLOAD_ENDPOINT = "$upload"
CONTROL_HINT_ENDPOINT = "$control_hint"
ALL_LATENCIES_KEY = "$all_latencies"
endpoint2method = {
    txt2img_sd_endpoint: "txt2img",
    img2img_sd_endpoint: "img2img",
    img2img_sr_endpoint: "sr",
    img2img_sod_endpoint: "sod",
    img2img_inpainting_endpoint: "inpainting",
    txt2img_sd_inpainting_endpoint: "sd_inpainting",
    txt2img_sd_outpainting_endpoint: "sd_outpainting",
    img2txt_caption_endpoint: "image_captioning",
    new_control_multi_endpoint: "run_multi_controlnet",
    img2img_harmonization_endpoint: "harmonization",
    paste_pipeline_endpoint: "paste_pipeline",
    UPLOAD_ENDPOINT: "get_image",
    CONTROL_HINT_ENDPOINT: "get_control_hint",
}


class APIs:
    algorithms: Dict[str, IAlgorithm]

    def __init__(
        self,
        *,
        algorithms: Optional[Dict[str, IAlgorithm]] = None,
        focuses_endpoints: Optional[List[str]] = None,
    ) -> None:
        if focuses_endpoints is None:
            focuses = None
        else:
            focuses = list(map(endpoint2algorithm, focuses_endpoints))
        OPT["verbose"] = focuses is not None
        OPT["lazy_load"] = True

        self._http_client = HttpClient()
        clients = dict(http=self._http_client, triton=None)
        if algorithms is not None:
            self.algorithms = algorithms
        else:
            self.algorithms = {
                k: v(clients)
                for k, v in registered_algorithms.items()
                if focuses is None or k in focuses
            }
            for v in self.algorithms.values():
                v.initialize()
        self._http_client.start()

    # lifecycle

    async def destroy(self) -> None:
        await self._http_client.stop()

    # algorithms

    async def _run(
        self, data: BaseModel, endpoint: str, **kw: Any
    ) -> List[Image.Image]:
        if isinstance(data, ReturnArraysModel):
            data.return_arrays = True
        task = endpoint2algorithm(endpoint)
        arrays = await self.algorithms[task].run(data, **kw)
        return list(map(Image.fromarray, arrays))

    async def txt2img(self, data: Txt2ImgSDModel, **kw: Any) -> List[Image.Image]:
        return await self._run(data, txt2img_sd_endpoint, **kw)

    async def img2img(self, data: Img2ImgSDModel, **kw: Any) -> List[Image.Image]:
        return await self._run(data, img2img_sd_endpoint, **kw)

    async def sr(self, data: Img2ImgSRModel, **kw: Any) -> List[Image.Image]:
        return await self._run(data, img2img_sr_endpoint, **kw)

    async def sod(self, data: Img2ImgSODModel, **kw: Any) -> List[Image.Image]:
        return await self._run(data, img2img_sod_endpoint, **kw)

    async def inpainting(
        self, data: Img2ImgInpaintingModel, **kw: Any
    ) -> List[Image.Image]:
        return await self._run(data, img2img_inpainting_endpoint, **kw)

    async def sd_inpainting(
        self, data: Txt2ImgSDInpaintingModel, **kw: Any
    ) -> List[Image.Image]:
        return await self._run(data, txt2img_sd_inpainting_endpoint, **kw)

    async def sd_outpainting(
        self, data: Txt2ImgSDOutpaintingModel, **kw: Any
    ) -> List[Image.Image]:
        return await self._run(data, txt2img_sd_outpainting_endpoint, **kw)

    async def image_captioning(self, data: Img2TxtModel, **kw: Any) -> List[str]:
        task = endpoint2algorithm(img2txt_caption_endpoint)
        result: TextModel = await self.algorithms[task].run(data, **kw)
        return [result.text]

    async def run_multi_controlnet(
        self, data: ControlMultiModel, **kw: Any
    ) -> List[Image.Image]:
        return await self._run(data, new_control_multi_endpoint, **kw)

    async def harmonization(
        self, data: Img2ImgHarmonizationModel, **kw: Any
    ) -> List[Image.Image]:
        return await self._run(data, img2img_harmonization_endpoint, **kw)

    async def paste_pipeline(
        self, data: PastePipelineModel, **kw: Any
    ) -> List[Image.Image]:
        return await self._run(data, paste_pipeline_endpoint, **kw)

    # special

    async def get_image(self, data: ImageModel, **kw: Any) -> List[Image.Image]:
        image = await download_image_with_retry(self._http_client.session, data.url)
        return [image]

    async def get_control_hint(
        self, hint_type: ControlNetHints, data: Dict[str, Any], **kw: Any
    ) -> List[Image.Image]:
        data = control_hint2hint_data_models[hint_type](**data)
        endpoint = control_hint2hint_endpoints[hint_type]
        return await self._run(data, endpoint, **kw)

    # workflow

    async def execute(
        self,
        workflow: Workflow,
        target: str,
        caches: Optional[Union[OrderedDict, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, TRes]:
        def _inject(k: str, ki_cache: TRes, current_node_data: dict) -> None:
            k_split = k.split(".")
            k0 = k_split[0]
            v0 = current_node_data.get(k0)
            if isinstance(v0, dict):
                _inject(".".join(k_split[1:]), ki_cache, v0)
            elif isinstance(v0, str) or v0 is None:
                if len(k_split) == 1:
                    if isinstance(ki_cache, Image.Image):
                        # we assign random_hash here to make mapping possible
                        current_node_data[k0] = random_hash()
                    else:
                        current_node_data[k0] = ki_cache
                else:
                    raise ValueError(
                        f"field under '{k0}' is already a vanilla string, "
                        f"but further keys are given: '{k_split[1:]}'"
                    )
            elif isinstance(v0, list):
                try:
                    k1 = int(k_split[1])
                except Exception:
                    raise ValueError(f"expected int key for '{k0}', got '{k_split[1]}'")
                v1 = v0[k1]
                if isinstance(v1, dict):
                    _inject(".".join(k_split[2:]), ki_cache, v1)
                elif len(k_split) == 2:
                    if isinstance(ki_cache, Image.Image):
                        # we assign random_hash here to make mapping possible
                        v0[k1] = random_hash()
                    else:
                        v0[k1] = ki_cache
                else:
                    raise ValueError(
                        f"list under '{k0}' is already a vanilla list, "
                        f"but further keys are given: '{k_split[2:]}'"
                    )
            else:
                raise ValueError(
                    f"field under '{k0}' should be one of (BaseModel, str, list), "
                    f"but got '{type(v0)}'"
                )

        all_latencies = {}
        if caches is None:
            caches = OrderedDict()
        else:
            caches = OrderedDict(caches)
        for layer in workflow.get_dependency_path(target).hierarchy:
            for item in layer:
                if item.key in caches:
                    continue
                node = item.data
                node_kw = shallow_copy_dict(kwargs)
                node_data = shallow_copy_dict(node.data)
                for k, k_packs in node.injections.items():
                    if not isinstance(k_packs, list):
                        k_packs = [k_packs]
                    for k_pack in k_packs:
                        ki_cache = caches[k][k_pack.index]
                        _inject(k_pack.field, ki_cache, node_data)
                        if isinstance(ki_cache, Image.Image):
                            node_kw[k_pack.field] = ki_cache
                endpoint = node.endpoint
                method_fn = getattr(self, endpoint2method[endpoint])
                t = time.time()
                if endpoint == UPLOAD_ENDPOINT:
                    data_model = ImageModel(**node_data)
                    item_res = await method_fn(data_model, **node_kw)
                elif endpoint == CONTROL_HINT_ENDPOINT:
                    hint_type = node_data.pop("hint_type")
                    item_res = await method_fn(hint_type, node_data, **node_kw)
                    endpoint = control_hint2hint_endpoints[hint_type]
                else:
                    data_model = self.get_data_model(endpoint, node_data)
                    item_res = await method_fn(data_model, **node_kw)
                if endpoint == UPLOAD_ENDPOINT:
                    ls = dict(download=time.time() - t)
                else:
                    ls = self.algorithms[endpoint2algorithm(endpoint)].last_latencies
                caches[item.key] = item_res
                all_latencies[item.key] = ls
        caches[ALL_LATENCIES_KEY] = all_latencies
        return caches

    # misc

    def get_data_model(self, endpoint: str, data: Dict[str, Any]) -> BaseModel:
        task = endpoint2algorithm(endpoint)
        return self.algorithms[task].model_class(**data)


__all__ = [
    "UPLOAD_ENDPOINT",
    "CONTROL_HINT_ENDPOINT",
    "ALL_LATENCIES_KEY",
    "APIs",
    "HighresModel",
    "Img2TxtModel",
    "Txt2ImgSDModel",
    "Img2ImgSDModel",
    "Img2ImgSRModel",
    "Img2ImgSODModel",
    "Img2ImgInpaintingModel",
    "Txt2ImgSDInpaintingModel",
    "Txt2ImgSDOutpaintingModel",
    "ControlNetHints",
    "ControlMultiModel",
    "Img2ImgHarmonizationModel",
]
