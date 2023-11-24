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

from obs import *
from PIL import Image
from kafka import KafkaConsumer
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Callable
from fastapi import Response
from pydantic import BaseModel

from cfclient.models import *
from cftool.misc import get_err_msg
from cftool.misc import shallow_copy_dict
from cfclient.core import HttpClient
from cfclient.core import TritonClient
from cfclient.utils import run_algorithm
from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkmoderation.v2 import ModerationClient
from huaweicloudsdkmoderation.v2.region.moderation_region import ModerationRegion

# This is necessary to register the algorithms
from cfcreator import *
from cfcreator.legacy.control_multi import ControlMultiModel as LegacyControlMultiModel

from producer import dump_queue
from producer import check_timeout
from producer import StatusData


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
cos_client = ObsClient(access_key_id=AK, secret_access_key=SK, server=SERVER)
image_mod_cred = BasicCredentials(AK, SK)
image_mod_client = (
    ModerationClient.new_builder()
    .with_credentials(image_mod_cred)
    .with_region(ModerationRegion.value_of(REGION))
    .build()
)
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
            print(f">>>>> post callback succeeded ({await res.json()})")

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
                    if vv is None:
                        continue
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


def get_upload_results(raw: List[Any], upload_fn: Callable) -> List[Any]:
    def _upload(v: List[Any]) -> List[Any]:
        return [
            _upload(elem)
            if isinstance(elem, list)
            else upload_fn(cos_client, elem)
            if isinstance(elem, np.ndarray)
            else upload_fn(cos_client, np.array(elem))
            if isinstance(elem, Image.Image)
            else elem
            for elem in v
        ]

    return _upload(raw)


def extract_urls(results: List[Union[UploadResponse, Any]], attr: str) -> List[Any]:
    def _extract(v: List[Any]) -> List[Any]:
        return [
            _extract(elem)
            if isinstance(elem, list)
            else getattr(elem, attr)
            if isinstance(elem, UploadResponse)
            else elem
            for elem in v
        ]

    return _extract(results)


# return (urls, reasons)
def audit_urls(
    model: BaseModel,
    url_results: List[Union[UploadResponse, Any]],
) -> Tuple[List[str], List[str]]:
    urls = extract_urls(url_results, "cdn")
    if isinstance(model, UseAuditModel) and not model.use_audit:
        reasons = [""] * len(url_results)
    else:
        reasons = []
        for i, rs in enumerate(url_results):
            if not isinstance(rs, UploadResponse):
                reasons.append("")
                continue
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


def consume_uid_from_queue(uid: str) -> None:
    queue = get_pending_queue()
    if uid in queue:
        queue.remove(uid)
        dump_queue(queue)


def echo_health() -> None:
    with open("/tmp/health", "w"):
        pass


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
    echo_health()
    # main loop
    try:
        for message in kafka_consumer:
            echo_health()
            data = json.loads(message.value)
            uid = data["uid"]
            task = data["task"]
            if task == "$health-check$":
                print(">>> incoming", data)
                redis_client.set(
                    uid,
                    json.dumps(
                        dict(status=Status.FINISHED, data=data),
                        ensure_ascii=False,
                    ),
                )
                consume_uid_from_queue(uid)
                continue
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
            if existing is not None and check_timeout(uid, StatusData(**existing)):
                print("!!! timeout", uid)
                continue
            echo_health()
            print(">>> working", uid)
            data = {} if existing is None else (existing.get("data", {}) or {})
            start_time = time.time()
            data["start_time"] = start_time
            create_time = data.get("create_time", start_time)
            redis_client.set(uid, json.dumps(dict(status=Status.WORKING, data=data)))
            # maintain queue
            echo_health()
            queue = get_pending_queue()
            if uid not in queue:
                queue.insert(0, uid)
                dump_queue(queue)
            procedure = "start"
            model = None
            try:
                algorithm = loaded_algorithms[task]
                model = algorithm.model_class(**params)  # type: ignore
                procedure = "start -> run_algorithm"
                if isinstance(model, ReturnArraysModel):
                    model.return_arrays = True
                res: Union[Response, Any] = await run_algorithm(algorithm, model)
                echo_health()
                latencies = algorithm.last_latencies
                t1 = time.time()
                if isinstance(algorithm, WorkflowAlgorithm):
                    intermediate = {}
                    response = dict(
                        is_exception=res.pop(WORKFLOW_IS_EXCEPTION_KEY, False),
                        intermediate=intermediate,
                    )
                    result = dict(uid=uid, response=response)
                    all_results = {}
                    save_to_private = (
                        not isinstance(model, WorkflowModel)
                        or model.save_intermediate_to_private
                    )
                    for k, v in res.items():
                        procedure = f"[{k}] run_algorithm -> upload_temp_image"
                        upload_fn = (
                            upload_temp_image
                            if k == WORKFLOW_TARGET_RESPONSE_KEY or not save_to_private
                            else upload_private_image
                        )
                        all_results[k] = get_upload_results(v, upload_fn)
                    t2 = time.time()
                    for k, k_results in all_results.items():
                        procedure = f"[{k}] upload_temp_image -> audit_image"
                        if save_to_private and k != WORKFLOW_TARGET_RESPONSE_KEY:
                            intermediate[k] = dict(
                                urls=extract_urls(k_results, "cos"),
                                reasons=[""] * len(k_results),
                            )
                        else:
                            k_urls, k_reasons = audit_urls(model, k_results)
                            if k != WORKFLOW_TARGET_RESPONSE_KEY:
                                intermediate[k] = dict(urls=k_urls, reasons=k_reasons)
                            else:
                                response["urls"] = k_urls
                                response["reasons"] = k_reasons
                    t3 = time.time()
                    procedure = "audit_image -> redis"
                elif task.startswith("control") or task.startswith("pipeline"):
                    procedure = "run_algorithm -> upload_temp_image"
                    url_results = get_upload_results(res, upload_temp_image)
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
                    elif task == "pipeline.paste" or task == "pipeline.product_sr":
                        result = dict(
                            uid=uid,
                            response=dict(url=urls[0], reason=reasons[0]),
                        )
                    elif task == "pipeline.product":
                        result = dict(
                            uid=uid,
                            response=dict(
                                raw_url=urls[0],
                                raw_reason=reasons[0],
                                depth_url=urls[1],
                                depth_reason=reasons[1],
                                result_urls=urls[2:],
                                result_reasons=reasons[2:],
                            ),
                        )
                    elif task.startswith("pipeline"):
                        num_hints = len(params["types"])
                        result = dict(
                            uid=uid,
                            response=dict(
                                raw_url=urls[0],
                                raw_reason=reasons[0],
                                hint_urls=urls[1 : num_hints + 1],
                                hint_reasons=reasons[1 : num_hints + 1],
                                result_urls=urls[num_hints + 1 :],
                                result_reasons=reasons[num_hints + 1 :],
                            ),
                        )
                    else:
                        raise ValueError(f"unrecognized task '{task}' occurred")
                elif algorithm.response_model_class is not None:
                    t2 = t3 = t1
                    procedure = "run_algorithm -> redis"
                    result = dict(uid=uid, response=res.dict())
                else:
                    procedure = "run_algorithm -> upload_temp_image"
                    if isinstance(res, list) and not res:
                        t2 = t3 = time.time()
                        result = dict(uid=uid, cdn="", cos="", safe=True, reason="")
                    else:
                        content = res[0] if isinstance(res, list) else res.body
                        urls = upload_temp_image(cos_client, content)
                        t2 = time.time()
                        procedure = "upload_temp_image -> audit_image"
                        if task != "img2img.sr":
                            try:
                                audit = audit_image(
                                    cos_client, image_mod_client, urls.path
                                )
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
                echo_health()
                procedure = "redis -> callback"
                await post_callback(callback_url, uid, True, result)
                echo_health()
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
                consume_uid_from_queue(uid)
                echo_health()
            except Exception as err:
                echo_health()
                end_time = time.time()
                torch.cuda.empty_cache()
                reason = f"{task} -> {procedure} : {get_err_msg(err)}"
                logging.exception(reason)
                data["uid"] = uid
                data["reason"] = reason
                data["end_time"] = end_time
                data["duration"] = end_time - create_time
                if model is None:
                    data["request"] = dict(task=task, params=simplify(params))
                else:
                    data["request"] = dict(task=task, model=simplify(model.dict()))
                echo_health()
                redis_client.set(
                    uid,
                    json.dumps(
                        dict(status=Status.EXCEPTION, data=data),
                        ensure_ascii=False,
                    ),
                )
                await post_callback(callback_url, uid, False, data)
                echo_health()
    finally:
        # clean up
        await http_client.stop()
        print(">>> end")


asyncio.run(consume())
