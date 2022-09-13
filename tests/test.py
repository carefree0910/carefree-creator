import asyncio

import numpy as np

from typing import Any
from typing import Dict
from aiohttp import ClientSession

from cfclient.utils import post


host = "http://localhost:8123"


def main(uri: str, data: Dict[str, Any]) -> Any:
    async def _run() -> Any:
        async with ClientSession() as session:
            return await post(f"{host}{uri}", data, session)

    return asyncio.run(_run())


if __name__ == "__main__":
    print(
        main(
            uri="/demo/hello",
            data=dict(
                name="carefree0910",
            ),
        )
    )
