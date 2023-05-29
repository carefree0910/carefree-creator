# `apis` sdk is used to prgrammatically call the algorithms.


from cfcreator import *
from cfclient.models import *
from PIL import Image
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from pydantic import Field
from pydantic import BaseModel
from collections import OrderedDict
from cftool.misc import random_hash
from cftool.misc import shallow_copy_dict
from cftool.data_structures import Item
from cftool.data_structures import Bundle
from cfclient.core import HttpClient
from cflearn.api.cv.diffusion import ControlNetHints


class WorkNode(BaseModel):
    key: str = Field(
        ...,
        description="Key of the node, should be identical within the same workflow",
    )
    endpoint: str = Field(..., description="Algorithm endpoint of the node")
    injections: Dict[Tuple[str, int], str] = Field(
        ...,
        description=(
            "Injection map, maps 'key' & 'index' from other `WorkNode` to "
            "'property' of the algorithm's field. In runtime, we'll collect "
            "the (list of) results from the depedencies (other `WorkNode`) and "
            "inject the specific result (based on 'index') to the algorithm's field.\n"
            "> If external caches is provided, the 'key' could be the key of the external cache.\n"
            "> Hierarchy injection is also supported, you just need to set 'property' to:\n"
            ">> `a.b.c` to inject the result to data['a']['b']['c']\n"
            ">> `a.0.b` to inject the first result to data['a'][0]['b']\n"
        ),
    )
    data: Dict[str, Any] = Field(..., description="Algorithm's data")

    def to_item(self) -> Item["WorkNode"]:
        return Item(self.key, self)


class Workflow(Bundle[WorkNode]):
    def push(self, node: WorkNode) -> None:
        return super().push(node.to_item())

    def toposort(self) -> List[List[Item[WorkNode]]]:
        out_degrees = {item.key: 0 for item in self}
        in_edges = {item.key: [] for item in self}
        for item in self:
            for dep, _ in item.data.injections:
                out_degrees[item.key] += 1
                in_edges[dep].append(item.key)

        ready = [k for k, v in out_degrees.items() if v == 0]
        result = []
        while ready:
            batch = ready.copy()
            result.append(batch)
            ready.clear()
            for dep in batch:
                for node in in_edges[dep]:
                    out_degrees[node] -= 1
                    if out_degrees[node] == 0:
                        ready.append(node)

        if len(self) != sum(map(len, result)):
            raise ValueError("cyclic dependency detected")

        return [list(map(self.get, batch)) for batch in result]

    def get_dependency_path(self, target: str) -> List[List[Item[WorkNode]]]:
        def dfs(key: str) -> None:
            if key in reachable:
                return
            reachable.add(key)
            for dep_key, _ in self.get(key).data.injections:
                dfs(dep_key)

        reachable = set()
        dfs(target)
        raw_result = self.toposort()
        result = []
        for raw_batch in raw_result:
            batch = []
            for item in raw_batch:
                if item.key in reachable:
                    batch.append(item)
            if batch:
                result.append(batch)
        return result


TRes = Union[List[str], List[Image.Image]]
CONTROL_HINT_ENDPOINT = "$control_hint"
endpoint2method = {
    txt2img_sd_endpoint: "txt2img",
    img2img_sd_endpoint: "img2img",
    img2img_sr_endpoint: "sr",
    img2img_sod_endpoint: "sod",
    img2img_inpainting_endpoint: "inpainting",
    txt2img_sd_inpainting_endpoint: "sd_inpainting",
    txt2img_sd_outpainting_endpoint: "sd_outpainting",
    img2txt_caption_endpoint: "image_captioning",
    CONTROL_HINT_ENDPOINT: "get_control_hint",
    new_control_multi_endpoint: "run_multi_controlnet",
    img2img_harmonization_endpoint: "harmonization",
}


class APIs:
    algorithms: Dict[str, IAlgorithm]

    def __init__(self, *, focuses_endpoints: Optional[List[str]] = None) -> None:
        if focuses_endpoints is None:
            focuses = None
        else:
            focuses = list(map(endpoint2algorithm, focuses_endpoints))
        OPT["verbose"] = focuses is not None
        OPT["lazy_load"] = True

        self._http_client = HttpClient()
        clients = dict(http=self._http_client, triton=None)
        self.algorithms = {
            k: v(clients)
            for k, v in algorithms.items()
            if focuses is None or k in focuses
        }
        self._http_client.start()
        for v in self.algorithms.values():
            v.initialize()

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

    async def get_control_hint(
        self, hint_type: ControlNetHints, **kw: Any
    ) -> List[Image.Image]:
        data = control_hint2hint_data_models[hint_type](**kw)
        endpoint = control_hint2hint_endpoints[hint_type]
        return await self._run(data, endpoint, **kw)

    async def run_multi_controlnet(
        self, data: ControlMultiModel, **kw: Any
    ) -> List[Image.Image]:
        return await self._run(data, new_control_multi_endpoint, **kw)

    async def harmonization(
        self, data: Img2ImgHarmonizationModel, **kw: Any
    ) -> List[Image.Image]:
        return await self._run(data, img2img_harmonization_endpoint, **kw)

    # workflow

    async def execute(
        self,
        workflow: Workflow,
        target: str,
        caches: Optional[OrderedDict] = None,
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

        if caches is None:
            caches = OrderedDict()
        for batch in workflow.get_dependency_path(target):
            for item in batch:
                if item.key in caches:
                    continue
                node = item.data
                node_kw = {}
                node_data = shallow_copy_dict(node.data)
                for (k, i), data_key in node.injections.items():
                    ki_cache = caches[k][i]
                    _inject(data_key, ki_cache, node_data)
                    if isinstance(ki_cache, Image.Image):
                        node_kw[data_key] = ki_cache
                endpoint = node.endpoint
                method_fn = getattr(self, endpoint2method[endpoint])
                if endpoint == CONTROL_HINT_ENDPOINT:
                    node_data.update(node_kw)
                    item_res = await method_fn(**node_data)
                else:
                    data_model = self.get_data_model(endpoint, node_data)
                    item_res = await method_fn(data_model, **node_kw)
                caches[item.key] = item_res
        return caches

    # misc

    def get_data_model(self, endpoint: str, data: Dict[str, Any]) -> BaseModel:
        task = endpoint2algorithm(endpoint)
        return self.algorithms[task].model_class(**data)


__all__ = [
    "WorkNode",
    "Workflow",
    "CONTROL_HINT_ENDPOINT",
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
