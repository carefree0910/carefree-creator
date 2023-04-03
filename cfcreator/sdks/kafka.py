# `kafka` sdk is used to access to the server launched by `uvicorn apis.kafka.producer:app --host 0.0.0.0 --port 8123`

import json
import time

from typing import Any
from typing import Dict
from typing import Callable
from typing import Optional
from cfclient.utils import get
from cfclient.utils import post
from cfcreator.common import endpoint2algorithm
from cfcreator.common import Status

from .utils import *


TCallback = Callable[[Dict[str, Any]], None]


async def push(host: str, endpoint: str, params: Dict[str, Any]) -> str:
    url = f"{host}/push"
    data = dict(task=endpoint2algorithm(endpoint), params=params)
    async with get_http_session() as session:
        res = await post(url, data, session)
        return res["uid"]


async def poll(
    host: str,
    uid: str,
    *,
    callback: Optional[TCallback] = None,
) -> Dict[str, Any]:
    url = f"{host}/status/{uid}"
    async with get_http_session() as session:
        while True:
            res = json.loads(await get(url, session))
            if callback is not None:
                callback(res)
            if res["status"] == Status.PENDING:
                time.sleep(res["pending"])
            elif res["status"] == Status.WORKING:
                time.sleep(1)
            else:
                return res


__all__ = [
    "push",
    "poll",
]
