import os
import json
import time
import yaml
import redis
import torch
import asyncio
import datetime
import logging.config

import numpy as np

from kafka import KafkaConsumer
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union
from fastapi import Response
from pydantic import BaseModel
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client

from cfclient.models import *
from cftool.misc import get_err_msg
from cftool.misc import shallow_copy_dict
from cfclient.core import HttpClient
from cfclient.core import TritonClient
from cfclient.utils import run_algorithm

# This is necessary to register the algorithms
from cfcreator import *
from cfcreator.legacy.control import ControlNetModel as LegacyControlNetModel
from cfcreator.legacy.control_multi import ControlMultiModel as LegacyControlMultiModel


# logging
root = os.path.dirname(__file__)
logging_root = os.path.join(root, "logs", "consumer")
os.makedirs(logging_root, exist_ok=True)
with open(os.path.join(root, "config.yml")) as f:
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S-%f")
    log_path = os.path.join(logging_root, f"{timestamp}.log")
    config = yaml.load(f, Loader=yaml.FullLoader)
    config["handlers"]["file"]["filename"] = log_path
    logging.config.dictConfig(config)

logging.getLogger("aiokafka").disabled = True
logging.getLogger("kafka.conn").disabled = True
logging.getLogger("kafka.client").disabled = True
logging.getLogger("kafka.cluster").disabled = True
logging.getLogger("kafka.coordinator").disabled = True
logging.getLogger("kafka.coordinator.consumer").disabled = True
logging.getLogger("kafka.metrics.metrics").disabled = True
logging.getLogger("kafka.protocol.parser").disabled = True
logging.getLogger("kafka.consumer.fetcher").disabled = True
logging.getLogger("kafka.consumer.group_coordinator").disabled = True
logging.getLogger("kafka.consumer.subscription_state").disabled = True


constants = dict(
    triton_host=None,
    triton_port=8000,
)


# clients
## cos client
config = CosConfig(
    Region=REGION,
    SecretId=SECRET_ID,
    SecretKey=SECRET_KEY,
    Scheme=SCHEME,
)
cos_client = CosS3Client(config)
## http client
http_client = HttpClient()
## triton client
triton_host = constants["triton_host"]
if triton_host is None:
    triton_client = None
else:
    triton_client = TritonClient(url=f"{triton_host}:{constants['triton_port']}")
## collect
clients = dict(
    http=http_client,
    triton=triton_client,
)

redis_client = redis.Redis(**redis_kwargs())
audit_redis_client = redis.Redis(**audit_redis_kwargs())
pending_queue_key = get_pending_queue_key()


# algorithms
loaded_algorithms: Dict[str, IAlgorithm] = {
    k: v(clients) for k, v in algorithms.items()
}


def get_redis_number(key: str) -> int:
    data = redis_client.get(key)
    if data is None:
        return 0
    return int(data.decode())  # type: ignore


def get_pending_queue() -> list:
    data = redis_client.get(pending_queue_key)
    if data is None:
        return []
    return json.loads(data)


async def post_callback(
    url: str,
    uid: str,
    success: bool,
    data: Dict[str, Any],
    retry: int = 10,
    interval: int = 3,
) -> None:
    if not url:
        print(">>>>> no callback is required")
        return

    print(f">>>>> post callback to {url}")

    async def fn() -> None:
        cb_data = dict(uid=uid, success=success, data=data)
        async with http_client.session.post(url, json=cb_data, timeout=interval) as res:
            if not res.ok:
                raise ValueError(f"post callback failed ({res.status})")

    msg = ""
    for i in range(retry):
        try:
            await fn()
            if i > 0:
                logging.warning(f"succeeded after {i} retries ({msg})")
            return
        except Exception as err:
            msg = get_err_msg(err)
        time.sleep(interval)
    if msg:
        print(
            f"\n\n!!! post to callback_url ({url}) failed "
            f"(After {retry} retries) ({msg}) !!!\n\n"
        )


def simplify(params: Any) -> Any:
    def _core(p: Any) -> Any:
        if isinstance(p, list):
            return [_core(v) for v in p]
        if not isinstance(p, dict):
            return p
        p = shallow_copy_dict(p)
        for k, v in p.items():
            if k == "custom_embeddings":
                if v is None or not isinstance(v, dict):
                    continue
                simplified_embeddings = {}
                for vk, vv in v.items():
                    try:
                        vva = np.atleast_2d(vv)
                        if vva.shape[1] <= 6:
                            simplified_embeddings[vk] = vv
                        else:
                            vva = np.concatenate([vva[:, :3], vva[:, -3:]], axis=1)
                            simplified_embeddings[vk] = vva.tolist()
                    except Exception as err:
                        simplified_embeddings[vk] = get_err_msg(err)
                p[k] = simplified_embeddings
            elif isinstance(p, (list, dict)):
                p[k] = _core(v)
        return p

    return _core(params)


# return (urls, reasons)
def audit_urls(
    model: BaseModel,
    url_results: List[UploadImageResponse],
) -> Tuple[List[str], List[str]]:
    urls = [rs.cdn for rs in url_results]
    if (
        not isinstance(model, (ControlNetModel, LegacyControlNetModel))
        or not model.use_audit
    ):
        reasons = [""] * len(url_results)
    else:
        reasons = []
        for i, rs in enumerate(url_results):
            try:
                audit = audit_image(cos_client, image_mod_client, rs.path)
            except:
                audit = AuditResponse(safe=False, reason="unknown")
            if audit.safe:
                reasons.append("")
            else:
                urls[i] = ""
                reasons.append(audit.reason)
    return urls, reasons


# kafka & redis
async def consume() -> None:
    OPT["verbose"] = False
    OPT["lazy_load"] = True

    topic = kafka_topic()
    expire_seconds = 10 * 365 * 24 * 3600

    redis_client.expire(pending_queue_key, expire_seconds)
    # initialize
    http_client.start()
    for v in loaded_algorithms.values():
        if isinstance(v, IWrapperAlgorithm):
            v.algorithms = loaded_algorithms
        v.initialize()
    print("> Algorithms are Loaded!")
    kafka_consumer = KafkaConsumer(
        topic,
        group_id=kafka_group_id(),
        bootstrap_servers=kafka_server(),
        max_poll_records=kafka_max_poll_records(),
        max_poll_interval_ms=kafka_max_poll_interval_ms(),
    )
    # main loop
    try:
        for message in kafka_consumer:
            data = json.loads(message.value)
            uid = data["uid"]
            task = data["task"]
            if task == "scene-generation":
                task = "txt2img.sd.outpainting"
            params = data["params"]
            callback_url = params.get("callback_url", "")
            existing = redis_client.get(uid)
            if existing is not None:
                existing = json.loads(existing)
                print(">>> existing", existing)
                if existing["status"] in (
                    Status.FINISHED,
                    Status.EXCEPTION,
                    Status.INTERRUPTED,
                ):
                    continue
            print(">>> working", uid)
            data = {} if existing is None else (existing.get("data", {}) or {})
            start_time = time.time()
            data["start_time"] = start_time
            create_time = data.get("create_time", start_time)
            redis_client.set(uid, json.dumps(dict(status=Status.WORKING, data=data)))
            procedure = "start"
            model = None
            try:
                algorithm = loaded_algorithms[task]
                model = algorithm.model_class(**params)  # type: ignore
                procedure = "start -> run_algorithm"
                if isinstance(model, ReturnArraysModel):
                    model.return_arrays = True
                res: Union[Response, Any] = await run_algorithm(algorithm, model)
                latencies = algorithm.last_latencies
                t1 = time.time()
                if (
                    isinstance(algorithm, WorkflowAlgorithm)
                    and params.get("intermediate") is not None
                ):
                    intermediate = {}
                    response = dict(intermediate=intermediate)
                    result = dict(uid=uid, response=response)
                    all_results = {}
                    for k, v in res.items():
                        procedure = f"[{k}] run_algorithm -> upload_temp_image"
                        all_results[k] = [
                            upload_temp_image(cos_client, arr) for arr in v
                        ]
                    t2 = time.time()
                    for k, k_results in all_results.items():
                        procedure = f"[{k}] upload_temp_image -> audit_image"
                        k_urls, k_reasons = audit_urls(model, k_results)
                        if k != WORKFLOW_TARGET_RESPONSE_KEY:
                            intermediate[k] = dict(urls=k_urls, reasons=k_reasons)
                        else:
                            response["urls"] = k_urls
                            response["reasons"] = k_reasons
                    t3 = time.time()
                    procedure = "audit_image -> redis"
                elif (
                    (
                        isinstance(algorithm, WorkflowAlgorithm)
                        and params.get("intermediate") is None
                    )
                    or task.startswith("control")
                    or task.startswith("pipeline")
                ):
                    procedure = "run_algorithm -> upload_temp_image"
                    url_results = [upload_temp_image(cos_client, arr) for arr in res]
                    t2 = time.time()
                    procedure = "upload_temp_image -> audit_image"
                    urls, reasons = audit_urls(model, url_results)
                    t3 = time.time()
                    procedure = "audit_image -> redis"
                    if isinstance(algorithm, WorkflowAlgorithm):
                        result = dict(
                            uid=uid,
                            response=dict(urls=urls, reasons=reasons),
                        )
                    elif task.startswith("control"):
                        if isinstance(model, ControlMultiModel):
                            keys = set(get_bundle_key(b) for b in model.controls)
                            num_cond = len(keys)
                        elif isinstance(model, LegacyControlMultiModel):
                            num_cond = len(model.types)
                        else:
                            num_cond = 1
                        result = dict(
                            uid=uid,
                            response=dict(
                                hint_urls=urls[:num_cond],
                                hint_reasons=reasons[:num_cond],
                                result_urls=urls[num_cond:],
                                result_reasons=reasons[num_cond:],
                            ),
                        )
                    elif task == "pipeline.paste":
                        result = dict(
                            uid=uid,
                            response=dict(url=urls[0], reason=reasons[0]),
                        )
                    else:
                        raise ValueError(f"unrecognized task '{task}' occurred")
                elif algorithm.response_model_class is not None:
                    t2 = t3 = t1
                    procedure = "run_algorithm -> redis"
                    result = dict(uid=uid, response=res.dict())
                else:
                    procedure = "run_algorithm -> upload_temp_image"
                    content = res[0] if isinstance(res, list) else res.body
                    urls = upload_temp_image(cos_client, content)
                    t2 = time.time()
                    procedure = "upload_temp_image -> audit_image"
                    if task != "img2img.sr":
                        try:
                            audit = audit_image(audit_redis_client, urls.path)
                        except:
                            audit = AuditResponse(safe=False, reason="unknown")
                    else:
                        audit = AuditResponse(safe=True, reason="")
                    t3 = time.time()
                    procedure = "audit_image -> redis"
                    result = dict(
                        uid=uid,
                        cdn=urls.cdn if audit.safe else "",
                        cos=urls.cos if audit.safe else "",
                        safe=audit.safe,
                        reason=audit.reason,
                    )
                result.update(data)
                end_time = time.time()
                result["end_time"] = end_time
                result["duration"] = end_time - create_time
                result["elapsed_times"] = dict(
                    pending=start_time - create_time,
                    run_algorithm=t1 - start_time,
                    algorithm_latencies=latencies,
                    upload=t2 - t1,
                    audit=t3 - t2,
                )
                result["request"] = dict(task=task, model=simplify(model.dict()))
                redis_client.set(
                    uid,
                    json.dumps(dict(status=Status.FINISHED, data=result)),
                )
                t4 = time.time()
                procedure = "redis -> callback"
                await post_callback(callback_url, uid, True, result)
                t5 = time.time()
                procedure = "callback -> update_elapsed_times"
                result["elapsed_times"].update(
                    dict(
                        redis=t4 - t3,
                        callback=t5 - t4,
                    )
                )
                redis_client.set(
                    uid,
                    json.dumps(dict(status=Status.FINISHED, data=result)),
                )
                procedure = "done"
                # maintain queue
                queue = get_pending_queue()
                if uid in queue:
                    queue.remove(uid)
                    redis_client.set(pending_queue_key, json.dumps(queue))
            except Exception as err:
                end_time = time.time()
                torch.cuda.empty_cache()
                reason = f"{task} -> {procedure} : {get_err_msg(err)}"
                data["uid"] = uid
                data["reason"] = reason
                data["end_time"] = end_time
                data["duration"] = end_time - create_time
                if model is None:
                    data["request"] = dict(task=task, params=simplify(params))
                else:
                    data["request"] = dict(task=task, model=simplify(model.dict()))
                redis_client.set(
                    uid,
                    json.dumps(
                        dict(status=Status.EXCEPTION, data=data),
                        ensure_ascii=False,
                    ),
                )
                await post_callback(callback_url, uid, False, data)
    finally:
        # clean up
        await http_client.stop()
        print(">>> end")


asyncio.run(consume())
