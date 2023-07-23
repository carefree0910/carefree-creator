# `kafka` sdk is used to access to the server launched by `uvicorn apis.kafka.producer:app --host 0.0.0.0 --port 8123`

import json
import time
import requests

from typing import Any
from typing import Dict
from typing import Callable
from typing import Optional
from cftool.misc import get_err_msg
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
        while True:
            try:
                res = await post(url, data, session)
                return res["uid"]
            except Exception as err:
                print(f"error occurred when pushing: {get_err_msg(err)}")
                time.sleep(1)


def push_sync(host: str, endpoint: str, params: Dict[str, Any]) -> str:
    url = f"{host}/push"
    data = dict(task=endpoint2algorithm(endpoint), params=params)
    return requests.post(url, json=data).json()["uid"]


async def poll(
    host: str,
    uid: str,
    *,
    callback: Optional[TCallback] = None,
) -> Dict[str, Any]:
    url = f"{host}/status/{uid}"
    async with get_http_session() as session:
        while True:
            try:
                res = json.loads(await get(url, session))
            except Exception as err:
                print(f"error occurred when polling: {get_err_msg(err)}")
                time.sleep(1)
                continue
            if callback is not None:
                callback(res)
            if res["status"] == Status.PENDING:
                time.sleep(res["pending"])
            elif res["status"] == Status.WORKING:
                time.sleep(1)
            else:
                return res


def poll_sync(
    host: str,
    uid: str,
    *,
    callback: Optional[TCallback] = None,
) -> Dict[str, Any]:
    while True:
        res = requests.get(f"{host}/status/{uid}").json()
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
    "push_sync",
    "poll_sync",
]
