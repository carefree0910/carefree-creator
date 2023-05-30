# `apis` sdk is used to prgrammatically call the algorithms.


import io

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from cfcreator import *
from cfclient.models import *
from PIL import Image
from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from typing import NamedTuple
from pydantic import Field
from pydantic import BaseModel
from collections import OrderedDict
from cftool.misc import random_hash
from cftool.misc import shallow_copy_dict
from cftool.data_structures import Item
from cftool.data_structures import Bundle
from cfclient.core import HttpClient
from cflearn.api.cv.diffusion import ControlNetHints


class InjectionPack(BaseModel):
    index: int
    field: str


class WorkNode(BaseModel):
    key: str = Field(
        ...,
        description="Key of the node, should be identical within the same workflow",
    )
    endpoint: str = Field(..., description="Algorithm endpoint of the node")
    injections: Dict[str, InjectionPack] = Field(
        ...,
        description=(
            "Injection map, maps 'key' from other `WorkNode` (A) to 'index' of A's results  & "
            "'field' of the algorithm's field. In runtime, we'll collect "
            "the (list of) results from the depedencies (other `WorkNode`) and "
            "inject the specific result (based on 'index') to the algorithm's field.\n"
            "> If external caches is provided, the 'key' could be the key of the external cache.\n"
            "> Hierarchy injection is also supported, you just need to set 'field' to:\n"
            ">> `a.b.c` to inject the result to data['a']['b']['c']\n"
            ">> `a.0.b` to inject the first result to data['a'][0]['b']\n"
        ),
    )
    data: Dict[str, Any] = Field(..., description="Algorithm's data")

    def to_item(self) -> Item["WorkNode"]:
        return Item(self.key, self)


class ToposortResult(NamedTuple):
    in_edges: Dict[str, Set[str]]
    hierarchy: List[List[Item[WorkNode]]]
    edge_labels: Dict[Tuple[str, str], str]


class Workflow(Bundle[WorkNode]):
    def push(self, node: WorkNode) -> None:
        return super().push(node.to_item())

    def toposort(self) -> ToposortResult:
        in_edges = {item.key: set() for item in self}
        out_degrees = {item.key: 0 for item in self}
        edge_labels: Dict[Tuple[str, str], str] = {}
        for item in self:
            for dep, pack in item.data.injections.items():
                in_edges[dep].add(item.key)
                out_degrees[item.key] += 1
                edge_labels[(item.key, dep)] = pack.field

        ready = [k for k, v in out_degrees.items() if v == 0]
        result = []
        while ready:
            layer = ready.copy()
            result.append(layer)
            ready.clear()
            for dep in layer:
                for node in in_edges[dep]:
                    out_degrees[node] -= 1
                    if out_degrees[node] == 0:
                        ready.append(node)

        if len(self) != sum(map(len, result)):
            raise ValueError("cyclic dependency detected")

        hierarchy = [list(map(self.get, layer)) for layer in result]
        return ToposortResult(in_edges, hierarchy, edge_labels)

    def get_dependency_path(self, target: str) -> ToposortResult:
        def dfs(key: str) -> None:
            if key in reachable:
                return
            reachable.add(key)
            for dep_key in self.get(key).data.injections:
                dfs(dep_key)

        reachable = set()
        dfs(target)
        in_edges, raw_hierarchy, edge_labels = self.toposort()
        hierarchy = []
        for raw_layer in raw_hierarchy:
            layer = []
            for item in raw_layer:
                if item.key in reachable:
                    layer.append(item)
            if layer:
                hierarchy.append(layer)
        return ToposortResult(in_edges, hierarchy, edge_labels)

    def render(
        self,
        *,
        target: Optional[str] = None,
        fig_w_ratio: int = 4,
        fig_h_ratio: int = 3,
        dpi: int = 200,
        node_size: int = 2000,
        node_shape: str = "s",
        node_color: str = "lightblue",
        layout: str = "multipartite_layout",
    ) -> Image.Image:
        # setup graph
        G = nx.DiGraph()
        if target is None:
            target = self.last.key
        in_edges, hierarchy, edge_labels = self.get_dependency_path(target)
        # setup plt
        fig_w = max(fig_w_ratio * len(hierarchy), 8)
        fig_h = fig_h_ratio * max(map(len, hierarchy))
        plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        box = plt.gca().get_position()
        plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # map key to indices
        key2idx = {}
        for layer in hierarchy:
            for node in layer:
                key2idx[node.key] = len(key2idx)
        # add nodes
        for i, layer in enumerate(hierarchy):
            for node in layer:
                G.add_node(key2idx[node.key], subset=f"layer_{i}")
        # add edges
        for dep, links in in_edges.items():
            for link in links:
                label = edge_labels[(link, dep)]
                G.add_edge(key2idx[dep], key2idx[link], label=label)
        # calculate positions
        layout_fn = getattr(nx, layout, None)
        if layout_fn is None:
            raise ValueError(f"unknown layout: {layout}")
        pos = layout_fn(G)
        # draw the nodes
        nodes_styles = dict(
            node_size=node_size,
            node_shape=node_shape,
            node_color=node_color,
        )
        nx.draw_networkx_nodes(G, pos, **nodes_styles)
        node_labels_styles = dict(
            font_size=18,
        )
        nx.draw_networkx_labels(G, pos, **node_labels_styles)
        # draw the edges
        nx_edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edges(
            G,
            pos,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=16,
            node_size=nodes_styles["node_size"],
            node_shape=nodes_styles["node_shape"],
        )
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx_edge_labels)
        # draw captions
        patches = [
            mpatches.Patch(color=node_color, label=f"{idx}: {key}")
            for key, idx in key2idx.items()
        ]
        plt.legend(handles=patches, bbox_to_anchor=(1, 0.5), loc="center left")
        # render
        plt.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return Image.open(buf)

    def to_json(self) -> List[Dict[str, Any]]:
        return [node.data.dict() for node in self]

    @classmethod
    def from_json(cls, data: List[Dict[str, Any]]) -> "Workflow":
        workflow = cls()
        for json in data:
            workflow.push(WorkNode(**json))
        return workflow


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

        if caches is None:
            caches = OrderedDict()
        for layer in workflow.get_dependency_path(target).hierarchy:
            for item in layer:
                if item.key in caches:
                    continue
                node = item.data
                node_kw = shallow_copy_dict(kwargs)
                node_data = shallow_copy_dict(node.data)
                for k, k_pack in node.injections.items():
                    ki_cache = caches[k][k_pack.index]
                    _inject(k_pack.field, ki_cache, node_data)
                    if isinstance(ki_cache, Image.Image):
                        node_kw[k_pack.field] = ki_cache
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
    "InjectionPack",
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
