from io import BytesIO
from PIL import Image
from typing import Any
from typing import Dict
from typing import Union
from pydantic import BaseModel
from cfclient.core import HttpClient
from cfclient.core import ClientSession
from cflearn.parameters import OPT


def get_url(endpoint: str) -> str:
    return f"{OPT.host}{endpoint}"


class get_http_session:
    def __init__(self):
        self._http_client = None

    async def __aenter__(self) -> ClientSession:
        self._http_client = HttpClient()
        self._http_client.start()
        return self._http_client.session

    async def __aexit__(self, *args: Any) -> None:
        await self._http_client.stop()
        self._http_client = None


async def get_image_res(url: str, d: Union[BaseModel, Dict[str, Any]]) -> Image.Image:
    if isinstance(d, BaseModel):
        d = d.dict()
    async with get_http_session() as session:
        async with session.post(url, json=d) as response:
            return Image.open(BytesIO(await response.read()))


__all__ = [
    "get_url",
    "get_http_session",
    "get_image_res",
]
