from typing import Any
from typing import Dict
from fastapi import Response
from cftool.misc import shallow_copy_dict


OPT = dict(
    verbose=True,
    save_gpu_ram=False,
    use_cos=True,
    request_domain="localhost",
    redis_kwargs=dict(host="localhost", port=6379, db=0),
    audit_redis_kwargs=dict(host="172.17.16.7", port=6379, db=1),
    bypass_audit=False,
    kafka_server="172.17.16.8:9092",
    kafka_topic="creator",
    kafka_group_id="creator-consumer-1",
    pending_queue_key="KAFKA_PENDING_QUEUE",
)


def verbose() -> bool:
    return OPT["verbose"]


def save_gpu_ram() -> bool:
    return OPT["save_gpu_ram"]


def use_cos() -> bool:
    return OPT["use_cos"]


def inject_headers(response: Response) -> None:
    response.headers["X-Request-Domain"] = OPT["request_domain"]


def redis_kwargs() -> Dict[str, Any]:
    return shallow_copy_dict(OPT["redis_kwargs"])


def audit_redis_kwargs() -> Dict[str, Any]:
    return shallow_copy_dict(OPT["audit_redis_kwargs"])


def bypass_audit() -> bool:
    return OPT["bypass_audit"]


def kafka_server() -> str:
    return OPT["kafka_server"]


def kafka_topic() -> str:
    return OPT["kafka_topic"]


def kafka_group_id() -> str:
    return OPT["kafka_group_id"]


def get_pending_queue_key() -> str:
    return OPT["pending_queue_key"]


__all__ = [
    "OPT",
    "use_cos",
    "verbose",
    "save_gpu_ram",
    "inject_headers",
    "redis_kwargs",
    "audit_redis_kwargs",
    "kafka_server",
    "kafka_topic",
    "kafka_group_id",
    "get_pending_queue_key",
]
