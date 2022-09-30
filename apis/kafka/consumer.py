import os
import json
import yaml
import redis
import asyncio
import datetime
import logging.config

from kafka import KafkaConsumer
from typing import Dict
from fastapi import Response
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client

from cfclient.models import *
from cfclient.core import HttpClient
from cfclient.core import TritonClient
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
config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY)
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

redis_client = redis.Redis(host="localhost", port=6379, db=0)
pending_queue_key = "KAFKA_PENDING_QUEUE"


# algorithms
loaded_algorithms: Dict[str, AlgorithmBase] = {
    k: v(clients)
    for k, v in algorithms.items()
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


# kafka & redis
async def consume() -> None:
    topic = "creator"
    expire_seconds = 10 * 365 * 24 * 3600

    redis_client.expire(pending_queue_key, expire_seconds)
    # initialize
    http_client.start()
    for k, v in loaded_algorithms.items():
        v.initialize()
    kafka_consumer = KafkaConsumer(
        topic,
        group_id="creator-consumer-1",
        bootstrap_servers="172.17.16.8:9092",
    )
    # main loop
    try:
        for message in kafka_consumer:
            data = json.loads(message.value)
            uid = data["uid"]
            task = data["task"]
            params = data["params"]
            redis_client.set(uid, json.dumps(dict(status="working", data=None)))
            try:
                algorithm = loaded_algorithms[task]
                model = algorithm.model_class(**params)  # type: ignore
                res: Response = await run_algorithm(algorithm, model)
                urls = upload_temp_image(cos_client, res.body)
                result = dict(cdn=urls.cdn, cos=urls.cos)
                redis_client.set(uid, json.dumps(dict(status="finished", data=result)))
            except Exception as err:
                redis_client.set(
                    uid,
                    json.dumps(dict(status="exception", data={"reason": str(err)})),
                )
            # maintain queue
            queue = get_pending_queue()
            if uid in queue:
                queue.remove(uid)
                redis_client.set(pending_queue_key, json.dumps(queue))
    finally:
        # clean up
        await http_client.stop()


asyncio.run(consume())
