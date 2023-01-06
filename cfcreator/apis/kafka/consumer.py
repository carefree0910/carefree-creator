import os
import json
import time
import yaml
import redis
import asyncio
import datetime
import logging.config

from kafka import KafkaConsumer
from typing import Any
from typing import Dict
from fastapi import Response
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client

from cfclient.models import *
from cfclient.core import HttpClient
from cfclient.core import TritonClient
from cfclient.utils import get_err_msg
from cfclient.utils import run_algorithm

# This is necessary to register the algorithms
from cfcreator import *


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
) -> None:
    if url:
        try:
            data = dict(uid=uid, success=success, data=data)
            async with http_client.session.post(url, json=data, timeout=1) as res:
                await res.json()
        except Exception as err:
            print(
                f"\n\n!!! post to callback_url ({url}) failed "
                f"({get_err_msg(err)}) !!!\n\n"
            )


# kafka & redis
async def consume() -> None:
    OPT["verbose"] = False

    topic = kafka_topic()
    expire_seconds = 10 * 365 * 24 * 3600

    redis_client.expire(pending_queue_key, expire_seconds)
    # initialize
    http_client.start()
    for k, v in loaded_algorithms.items():
        v.initialize()
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
                res: Response = await run_algorithm(algorithm, model)
                latencies = algorithm.last_latencies
                t1 = time.time()
                procedure = "run_algorithm -> upload_temp_image"
                urls = upload_temp_image(cos_client, res.body)
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
                result["request"] = dict(task=task, model=model.dict())
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
                reason = f"{task} -> {json.dumps(params, ensure_ascii=False)} -> {procedure} : {get_err_msg(err)}"
                data["uid"] = uid
                data["reason"] = reason
                data["end_time"] = end_time
                data["duration"] = end_time - create_time
                if model is None:
                    data["request"] = dict(task=task, params=params)
                else:
                    data["request"] = dict(task=task, model=model.dict())
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
