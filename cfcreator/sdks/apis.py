# `apis` sdk is used to prgrammatically call the algorithms.


from cfcreator import *
from cfclient.models import *
from PIL import Image
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from pydantic import BaseModel
from cfclient.core import HttpClient


class APIs:
    algorithms: Dict[str, IAlgorithm]

    def __init__(self, *, focuses_endpoints: Optional[List[str]] = None) -> None:
        if focuses_endpoints is None:
            focuses = None
        else:
            focuses = list(map(endpoint2algorithm, focuses_endpoints))
        OPT["verbose"] = focuses is not None
        OPT["lazy_load"] = True

        self._http_client = HttpClient()
        clients = dict(http=self._http_client, triton=None)
        self.algorithms = {
            k: v(clients)
            for k, v in algorithms.items()
            if focuses is None or k in focuses
        }
        self._http_client.start()
        for v in self.algorithms.values():
            v.initialize()

    # lifecycle

    async def destroy(self) -> None:
        await self._http_client.stop()

    # algorithms

    async def _run(
        self, data: BaseModel, endpoint: str, **kw: Any
    ) -> List[Image.Image]:
        if isinstance(data, ReturnArraysModel):
            data.return_arrays = True
        task = endpoint2algorithm(endpoint)
        arrays = await self.algorithms[task].run(data, **kw)
        return list(map(Image.fromarray, arrays))

    async def txt2img(self, data: Txt2ImgSDModel, **kw: Any) -> List[Image.Image]:
        return await self._run(data, txt2img_sd_endpoint, **kw)

    async def img2img(self, data: Img2ImgSDModel, **kw: Any) -> List[Image.Image]:
        return await self._run(data, img2img_sd_endpoint, **kw)

    async def sr(self, data: Img2ImgSRModel) -> List[Image.Image]:
        return await self._run(data, img2img_sr_endpoint)

    async def sod(self, data: Img2ImgSODModel) -> List[Image.Image]:
        return await self._run(data, img2img_sod_endpoint)

    async def inpainting(
        self, data: Img2ImgInpaintingModel, **kw: Any
    ) -> List[Image.Image]:
        return await self._run(data, img2img_inpainting_endpoint, **kw)

    async def sd_inpainting(
        self, data: Txt2ImgSDInpaintingModel, **kw: Any
    ) -> List[Image.Image]:
        return await self._run(data, txt2img_sd_inpainting_endpoint, **kw)

    async def sd_outpainting(
        self, data: Txt2ImgSDOutpaintingModel, **kw: Any
    ) -> List[Image.Image]:
        return await self._run(data, txt2img_sd_outpainting_endpoint, **kw)

    async def image_captioning(self, data: Img2TxtModel) -> List[str]:
        task = endpoint2algorithm(img2txt_caption_endpoint)
        result: TextModel = await self.algorithms[task].run(data)
        return [result.text]


__all__ = [
    "APIs",
    "HighresModel",
    "Img2TxtModel",
    "Txt2ImgSDModel",
    "Img2ImgSDModel",
    "Img2ImgSRModel",
    "Img2ImgSODModel",
    "Img2ImgInpaintingModel",
    "Txt2ImgSDInpaintingModel",
    "Txt2ImgSDOutpaintingModel",
]
