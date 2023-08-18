import time

import numpy as np

from PIL import Image
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from fastapi import Response
from pydantic import Field
from cftool.data_structures import WorkNode
from cftool.data_structures import Workflow

from .common import get_response
from .common import IWrapperAlgorithm
from .common import ReturnArraysModel


workflow_endpoint = "/workflow"
WORKFLOW_TARGET_RESPONSE_KEY = "$target"


class WorkflowModel(ReturnArraysModel):
    nodes: List[WorkNode] = Field(..., description="The nodes in the workflow.")
    target: str = Field(..., description="The target node.")
    intermediate: Optional[List[str]] = Field(
        None,
        description="The intermediate nodes that you want to collect results from.",
    )
    caches: Optional[Dict[str, Any]] = Field(None, description="The preset caches.")


@IWrapperAlgorithm.auto_register()
class WorkflowAlgorithm(IWrapperAlgorithm):
    model_class = WorkflowModel

    endpoint = workflow_endpoint

    async def run(self, data: WorkflowModel, *args: Any, **kwargs: Any) -> Response:
        self.log_endpoint(data)
        t0 = time.time()
        workflow = Workflow()
        for node in data.nodes:
            workflow.push(node)
        t1 = time.time()
        results = await self.apis.execute(workflow, data.target, data.caches)
        t2 = time.time()
        # fetch target
        target_result = results[data.target]
        if not target_result or isinstance(target_result[0], str):
            res = target_result
        else:
            arrays = list(map(np.array, target_result))
            res = get_response(data, arrays)
        # fetch intermediate
        if data.intermediate is not None:
            if not data.return_arrays:
                msg = "`return_arrays` should be True when `intermediate` is specified."
                raise ValueError(msg)
            res = {WORKFLOW_TARGET_RESPONSE_KEY: res}
            for key in data.intermediate:
                intermediate_result = results[key]
                if intermediate_result:
                    if isinstance(intermediate_result[0], Image.Image):
                        intermediate_result = list(map(np.array, intermediate_result))
                res[key] = intermediate_result
        latencies = {
            "get_workflow": t1 - t0,
            "inference": t2 - t1,
            "get_response": time.time() - t2,
        }
        self.log_times(latencies)
        self.last_latencies["inference_details"] = results[self.latencies_key]
        return res


__all__ = [
    "workflow_endpoint",
    "WORKFLOW_TARGET_RESPONSE_KEY",
    "WorkflowModel",
    "WorkflowAlgorithm",
]
