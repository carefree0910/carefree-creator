import os
import json
import time
import yaml
import redis
import datetime
import logging.config

from enum import Enum
from kafka import KafkaProducer
from kafka import KafkaAdminClient
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import NamedTuple
from fastapi import FastAPI
from fastapi import Response
from pydantic import Field
from pydantic import BaseModel
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
from pkg_resources import get_distribution
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware

from cftool.web import get_responses
from cftool.misc import get_err_msg
from cftool.misc import random_hash
from cftool.misc import shallow_copy_dict

from cfcreator import *


app = FastAPI()
root = os.path.dirname(__file__)

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# logging
logging_root = os.path.join(root, "logs", "producer")
os.makedirs(logging_root, exist_ok=True)
with open(os.path.join(root, "config.yml")) as f:
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S-%f")
    log_path = os.path.join(logging_root, f"{timestamp}.log")
    config = yaml.load(f, Loader=yaml.FullLoader)
    config["handlers"]["file"]["filename"] = log_path
    logging.config.dictConfig(config)

excluded_endpoints = {"/health", "/redoc", "/docs", "/openapi.json"}


class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not record.args:
            return False
        if len(record.args) < 3:
            return False
        if record.args[2] in excluded_endpoints:
            return False
        return True


logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
logging.getLogger("dicttoxml").disabled = True
logging.getLogger("kafka.conn").disabled = True
logging.getLogger("kafka.cluster").disabled = True
logging.getLogger("kafka.coordinator").disabled = True
logging.getLogger("kafka.consumer.subscription_state").disabled = True


# clients
config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY)
cos_client = CosS3Client(config)
redis_client = redis.Redis(**redis_kwargs())
kafka_admin = KafkaAdminClient(bootstrap_servers=kafka_server())
kafka_producer = KafkaProducer(bootstrap_servers=kafka_server())


# schema


DOCS_TITLE = "FastAPI client"
DOCS_VERSION = get_distribution("carefree-client").version
DOCS_DESCRIPTION = (
    "This is a client framework based on FastAPI. "
    "It also supports interacting with Triton Inference Server."
)


def carefree_schema() -> Dict[str, Any]:
    schema = get_openapi(
        title=DOCS_TITLE,
        version=DOCS_VERSION,
        description=DOCS_DESCRIPTION,
        contact={
            "name": "Get Help with this API",
            "email": "syameimaru.saki@gmail.com",
        },
        routes=app.routes,
    )
    app.openapi_schema = schema
    return app.openapi_schema


# health check


class HealthStatus(Enum):
    ALIVE = "alive"


class HealthCheckResponse(BaseModel):
    status: HealthStatus


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    return {"status": "alive"}


# get prompt


@app.post("/translate")
@app.post("/get_prompt")
def get_prompt(data: GetPromptModel, response: Response) -> GetPromptResponse:
    inject_headers(response)
    text = data.text
    audit = audit_text(cos_client, text)
    if not audit.safe:
        return GetPromptResponse(text="", success=False, reason=audit.reason)
    return GetPromptResponse(text=text, success=True, reason="")


# kafka & redis


pending_queue_key = get_pending_queue_key()


def get_redis_number(key: str) -> Optional[int]:
    data = redis_client.get(key)
    if data is None:
        return None
    return int(data.decode())  # type: ignore


class ProducerModel(BaseModel):
    task: str
    params: Dict[str, Any]
    notify_url: str = Field("", description="callback url to post to")


class ProducerResponse(BaseModel):
    uid: str
    msg: str
    success: bool


# 30min = timeout
queue_timeout_threshold = 30 * 60
## check timeout interval = 1min
queue_timeout_timer = time.time()
queue_timeout_check_interval = 60


def dump_queue(queue: List[str]) -> None:
    redis_client.set(pending_queue_key, json.dumps(queue))


def check_timeout(uid: str, data: "StatusData") -> bool:
    d = shallow_copy_dict(data.data or {})
    create_time = d.get("create_time", None)
    start_time = d.get("start_time", None)
    for t in [create_time, start_time]:
        if t is not None:
            dt = time.time() - t
            if dt >= queue_timeout_threshold:
                if data.status in (Status.PENDING, Status.WORKING):
                    d["reason"] = f"timeout after {dt}s"
                    redis_client.set(
                        uid,
                        json.dumps(dict(status=Status.EXCEPTION, data=d)),
                    )
                return True
    return False


def get_pending_queue() -> List[str]:
    data = redis_client.get(pending_queue_key)
    if data is None:
        return []
    return json.loads(data)


def get_clean_queue() -> List[str]:
    global queue_timeout_timer
    queue = get_pending_queue()
    queue_size = len(queue)
    # check timeout / finished / interrupted
    if time.time() - queue_timeout_timer >= queue_timeout_check_interval:
        queue_timeout_timer = time.time()
        clear_indices = []
        for i, uid in enumerate(queue):
            uid_pack = redis_client.get(uid)
            if uid_pack is None:
                continue
            uid_data = StatusData(**json.loads(uid_pack))
            if uid_data.status in (Status.FINISHED, Status.INTERRUPTED):
                clear_indices.append(i)
                continue
            if check_timeout(uid, uid_data):
                clear_indices.append(i)
        for idx in clear_indices[::-1]:
            queue.pop(idx)
    # check redundant
    exist = set()
    clear_indices = []
    for i, uid in enumerate(queue):
        if uid not in exist:
            exist.add(uid)
        else:
            clear_indices.append(i)
    for idx in clear_indices[::-1]:
        queue.pop(idx)
    # check consistency
    latest_queue = get_pending_queue()
    ## if pending queue is already updated, ignore the cleanups and return the latest queue
    if queue_size != len(latest_queue):
        queue = latest_queue
    ## else, update the pending queue with the cleaned queue
    else:
        dump_queue(queue)
    return queue


@app.post("/push", responses=get_responses(ProducerResponse))
async def push(data: ProducerModel, response: Response) -> ProducerResponse:
    inject_headers(response)
    new_uid = random_hash()
    # append new uid and dump
    queue = get_clean_queue()
    queue.append(new_uid)
    dump_queue(queue)
    redis_client.set(
        new_uid,
        json.dumps(dict(status=Status.PENDING, data=dict(create_time=time.time()))),
    )
    # send to kafka
    data.params["callback_url"] = data.notify_url
    try:
        kafka_producer.send(
            kafka_topic(),
            json.dumps(
                dict(
                    uid=new_uid,
                    task=data.task,
                    params=data.params,
                ),
                ensure_ascii=False,
            ).encode("utf-8"),
        )
    except Exception as err:
        msg = get_err_msg(err)
        logging.exception(msg)
        return ProducerResponse(uid=new_uid, msg=msg, success=False)
    return ProducerResponse(uid=new_uid, msg="", success=True)


class InterruptModel(BaseModel):
    uid_list: List[str]
    force_interrupt: bool = False


class InterruptSingleResponse(BaseModel):
    success: bool
    reason: str


class InterruptResponse(BaseModel):
    results: List[InterruptSingleResponse]


@app.post("/interrupt", responses=get_responses(InterruptResponse))
async def interrupt(data: InterruptModel, response: Response) -> InterruptResponse:
    inject_headers(response)
    results = []
    for uid in data.uid_list:
        existing = redis_client.get(uid)
        if existing is None:
            results.append(InterruptSingleResponse(success=False, reason="not found"))
            continue
        existing_data = StatusData(**json.loads(existing))
        if not data.force_interrupt and existing_data.status != Status.PENDING:
            results.append(
                InterruptSingleResponse(
                    success=False,
                    reason=f"not in pending status ({existing_data.status})",
                )
            )
            continue
        redis_client.set(
            uid,
            json.dumps(dict(status=Status.INTERRUPTED, data=existing_data.data)),
        )
        results.append(InterruptSingleResponse(success=True, reason=""))
    return InterruptResponse(results=results)


class ServerStatusResponse(BaseModel):
    is_ready: bool
    num_pending: int
    pending_queue: List[str]
    details: Dict[str, List[str]]


def get_statuses(queue: List[str]) -> List[str]:
    return [fetch_redis(uid).status for uid in queue]


def get_real_pending(queue: List[str], statuses: List[str]) -> List[str]:
    return [
        uid
        for uid, status in zip(queue, statuses)
        if status not in (Status.FINISHED, Status.EXCEPTION, Status.INTERRUPTED)
    ]


def get_real_lag(queue: List[str]) -> int:
    statuses = get_statuses(queue)
    return len(get_real_pending(queue, statuses))


@app.get("/server_status", responses=get_responses(ServerStatusResponse))
async def server_status(response: Response) -> ServerStatusResponse:
    inject_headers(response)
    members = kafka_admin.describe_consumer_groups([kafka_group_id()])[0].members
    queue = get_clean_queue()
    statuses = get_statuses(queue)
    real_pending = get_real_pending(queue, statuses)
    details: Dict[str, List[str]] = {}
    for uid, status in zip(queue, statuses):
        if status == Status.INTERRUPTED:
            continue
        details.setdefault(status, []).append(uid)
    return ServerStatusResponse(
        is_ready=len(members) > 0,
        num_pending=len(real_pending),
        pending_queue=real_pending,
        details=details,
    )


class StatusResponse(BaseModel):
    status: Status
    pending: int
    data: Optional[Any]


class StatusData(NamedTuple):
    status: Status
    data: Optional[Any]


def fetch_redis(uid: str) -> StatusData:
    data = redis_client.get(uid)
    if data is None:
        return StatusData(status=Status.NOT_FOUND, data=None)
    try:
        return StatusData(**json.loads(data))
    except Exception as e:
        return StatusData(status=Status.EXCEPTION, data=dict(reason=get_err_msg(e)))


def _get_status(uid: str) -> StatusResponse:
    record = fetch_redis(uid)
    if record.status != Status.PENDING:
        lag = 0
    else:
        queue = get_clean_queue()
        # add `uid` back to queue if it's not in the queue
        try:
            lag = get_real_lag(queue[: queue.index(uid)]) + 1
        except:
            lag = len(queue) + 1
            queue.append(uid)
            dump_queue(queue)
    return StatusResponse(status=record.status, data=record.data, pending=lag)


@app.get("/status/{uid}", responses=get_responses(StatusResponse))
async def get_status(uid: str, response: Response) -> StatusResponse:
    inject_headers(response)
    return _get_status(uid)


class BatchStatusModel(BaseModel):
    uid_list: List[str]


class BatchStatusResponse(BaseModel):
    results: List[StatusResponse]


@app.post("/batch_status", responses=get_responses(BatchStatusModel))
async def get_batch_status(
    data: BatchStatusModel,
    response: Response,
) -> BatchStatusResponse:
    inject_headers(response)
    return BatchStatusResponse(results=[_get_status(uid) for uid in data.uid_list])


# audit


class AuditCallbackModel(BaseModel):
    EventName: str
    JobsDetail: AuditJobsDetailModel


@app.post("/audit_callback")
async def audit_callback(data: AuditCallbackModel) -> None:
    key = data.JobsDetail.Object
    redis_client.expire(key, 3600)
    redis_client.set(
        key,
        json.dumps(data.JobsDetail.dict()),
    )


# events


@app.on_event("startup")
async def startup() -> None:
    pass


@app.on_event("shutdown")
async def shutdown() -> None:
    pass


# schema

app.openapi = carefree_schema


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("interface:app", host="0.0.0.0", port=8989, reload=True)
