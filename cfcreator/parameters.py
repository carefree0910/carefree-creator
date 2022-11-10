from typing import Any
from typing import Dict
from cftool.misc import shallow_copy_dict


OPT = dict(
    verbose=True,
    save_gpu_ram=False,
    use_cos=True,
    redis_kwargs=dict(host="localhost", port=6379, db=0),
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


def redis_kwargs() -> Dict[str, Any]:
    return shallow_copy_dict(OPT["redis_kwargs"])


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
    "redis_kwargs",
    "kafka_server",
    "kafka_topic",
    "kafka_group_id",
    "get_pending_queue_key",
]
