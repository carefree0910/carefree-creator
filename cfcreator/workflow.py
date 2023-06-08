import io
import time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image
from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from typing import NamedTuple
from fastapi import Response
from pydantic import Field
from pydantic import BaseModel
from cftool.data_structures import Item
from cftool.data_structures import Bundle

from .common import get_response
from .common import IAlgorithm
from .common import ReturnArraysModel


# data structures


class InjectionPack(BaseModel):
    index: int
    field: str


class WorkNode(BaseModel):
    key: str = Field(
        ...,
        description="Key of the node, should be identical within the same workflow",
    )
    endpoint: str = Field(..., description="Algorithm endpoint of the node")
    injections: Dict[str, Union[InjectionPack, List[InjectionPack]]] = Field(
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
            for dep, packs in item.data.injections.items():
                in_edges[dep].add(item.key)
                out_degrees[item.key] += 1
                if not isinstance(packs, list):
                    packs = [packs]
                for pack in packs:
                    label_key = (item.key, dep)
                    existing_label = edge_labels.get(label_key)
                    if existing_label is None:
                        edge_labels[label_key] = pack.field
                    else:
                        edge_labels[label_key] = f"{existing_label}, {pack.field}"

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
                if dep not in key2idx or link not in key2idx:
                    continue
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


# endpoint


workflow_endpoint = "/workflow"


class WorkflowModel(ReturnArraysModel):
    nodes: List[WorkNode] = Field(..., description="The nodes in the workflow.")
    target: str = Field(..., description="The target node.")
    caches: Optional[Dict[str, Any]] = Field(None, description="The preset caches.")


@IAlgorithm.auto_register()
class WorkflowAlgorithm(IAlgorithm):
    model_class = WorkflowModel

    algorithms: Optional[Dict[str, IAlgorithm]] = None
    last_workflow: Optional[Workflow] = None

    endpoint = workflow_endpoint

    def initialize(self) -> None:
        from cfcreator.sdks.apis import APIs
        from cfcreator.sdks.apis import ALL_LATENCIES_KEY

        if self.algorithms is None:
            raise ValueError("`algorithms` should be provided for `WorkflowAlgorithm`.")
        self.apis = APIs(algorithms=self.algorithms)
        self.latencies_key = ALL_LATENCIES_KEY

    async def run(self, data: WorkflowModel, *args: Any, **kwargs: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        workflow = Workflow()
        for node in data.nodes:
            workflow.push(node)
        self.last_workflow = workflow
        t1 = time.time()
        results = await self.apis.execute(workflow, data.target, data.caches)
        t2 = time.time()
        target_result = results[data.target]
        if isinstance(target_result[0], str):
            raise ValueError("The target node should return images.")
        arrays = list(map(np.array, target_result))
        t3 = time.time()
        res = get_response(data, arrays)
        latencies = {
            "get_workflow": t1 - t0,
            "inference": t2 - t1,
            "postprocess": t3 - t2,
            "get_response": time.time() - t3,
        }
        self.log_times(latencies)
        self.last_latencies["inference_details"] = results[self.latencies_key]
        return res


__all__ = [
    "InjectionPack",
    "WorkNode",
    "ToposortResult",
    "Workflow",
    "workflow_endpoint",
    "WorkflowModel",
    "WorkflowAlgorithm",
]
