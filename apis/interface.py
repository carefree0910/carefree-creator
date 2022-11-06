import os
import yaml
import torch
import cflearn
import logging
import datetime
import logging.config

from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from fastapi import FastAPI
from fastapi import Response
from pydantic import BaseModel
from pkg_resources import get_distribution
from cftool.array import tensor_dict_type
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware

from cfclient.models import *
from cfclient.core import HttpClient
from cfclient.core import TritonClient
from cfclient.utils import get_err_msg
from cfclient.utils import get_responses
from cfclient.utils import run_algorithm
from cfclient.utils import get_image_response_kwargs

from cfcreator import *


constants = dict(
    triton_host=None,
    triton_port=8000,
)

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

# models
model_root = os.path.join(root, "models")
token_root = os.path.join(root, "tokens")

# logging
logging_root = os.path.join(root, "logs")
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

# clients
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


# algorithms
all_algorithms: Dict[str, AlgorithmBase] = {
    k: v(clients) for k, v in algorithms.items()
}


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


# demo


@app.post(demo_hello_endpoint, responses=get_responses(HelloResponse))
async def hello(data: HelloModel) -> HelloResponse:
    return await run_algorithm(all_algorithms["demo.hello"], data)


# get prompt


@app.post("/translate")
@app.post("/get_prompt")
def get_prompt(data: GetPromptModel) -> GetPromptResponse:
    return GetPromptResponse(text=data.text, success=True, reason="")


# switch local checkpoint


class SwitchCheckpointModel(BaseModel):
    key: str
    model: str
    is_full_path: bool = False


class SwitchCheckpointResponse(BaseModel):
    success: bool
    reason: str


class AvailableVersions(BaseModel):
    versions: List[str]


@app.get("/available_versions")
def get_available_api_versions() -> AvailableVersions:
    return AvailableVersions(versions=available_apis())


@app.post("/switch")
def switch_checkpoint(data: SwitchCheckpointModel) -> SwitchCheckpointResponse:
    api = get_api(data.key)
    if api is None:
        return SwitchCheckpointResponse(
            success=False,
            reason=f"'{data.key}' is not a valid key, available keys are: {', '.join(available_apis())}",
        )
    if data.is_full_path:
        model_path = data.model
    else:
        model_path = os.path.join(model_root, data.model)
    if not os.path.isfile(model_path):
        return SwitchCheckpointResponse(
            success=False,
            reason=f"cannot find '{data.model}' under '{model_root}'",
        )
    try:
        cflearn.scripts.sd.convert(model_path, api, load=True)
        return SwitchCheckpointResponse(success=True, reason="")
    except Exception as err:
        logging.exception(err)
        return SwitchCheckpointResponse(success=False, reason=get_err_msg(err))


# inject custom tokens


custom_embeddings: tensor_dict_type = {}


def _inject_custom_tokens(root: str) -> tensor_dict_type:
    local_customs: tensor_dict_type = {}
    if not os.path.isdir(root):
        return local_customs
    for file in os.listdir(root):
        try:
            path = os.path.join(root, file)
            d = torch.load(path, map_location="cpu")
            local_customs.update({k: v.tolist() for k, v in d.items()})
        except:
            continue
    if local_customs:
        print(f"> Following tokens are loaded: {', '.join(sorted(local_customs))}")
        custom_embeddings.update(local_customs)
    return local_customs


class InjectCustomTokenModel(BaseModel):
    root: str


class InjectCustomTokenResponse(BaseModel):
    success: bool
    reason: str


@app.post("/inject_tokens")
def inject_custom_tokens(data: InjectCustomTokenModel) -> InjectCustomTokenResponse:
    if not _inject_custom_tokens(data.root):
        return InjectCustomTokenResponse(
            success=False,
            reason=f"cannot find any tokens under '{data.root}'",
        )
    return InjectCustomTokenResponse(success=True, reason="")


# meta


registered_algorithms = set()


def register_endpoint(endpoint: str, data_model: Type[BaseModel]) -> None:
    name = endpoint[1:].replace("/", "_")
    algorithm_name = endpoint2algorithm(endpoint)
    algorithm = all_algorithms[algorithm_name]
    registered_algorithms.add(algorithm_name)

    @app.post(endpoint, **get_image_response_kwargs(), name=name)
    async def _(data: data_model) -> Response:
        if isinstance(data, DiffusionModel):
            data.custom_embeddings = custom_embeddings
        return await run_algorithm(algorithm, data)


# txt2img
register_endpoint(txt2img_sd_endpoint, Txt2ImgSDModel)
register_endpoint(txt2img_sd_inpainting_endpoint, Txt2ImgSDInpaintingModel)
register_endpoint(txt2img_sd_outpainting_endpoint, Txt2ImgSDOutpaintingModel)

# img2img
register_endpoint(img2img_sd_endpoint, Img2ImgSDModel)
register_endpoint(img2img_sr_endpoint, Img2ImgSRModel)
register_endpoint(img2img_inpainting_endpoint, Img2ImgInpaintingModel)
register_endpoint(img2img_semantic2img_endpoint, Img2ImgSemantic2ImgModel)

# events


@app.on_event("startup")
async def startup() -> None:
    http_client.start()
    OPT["use_cos"] = False
    # OPT["save_gpu_ram"] = True
    for k, v in all_algorithms.items():
        if k in registered_algorithms:
            v.initialize()
    _inject_custom_tokens(token_root)
    print("> Server is Ready!")


@app.on_event("shutdown")
async def shutdown() -> None:
    await http_client.stop()


# schema

app.openapi = carefree_schema


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("interface:app", host="0.0.0.0", port=8989, reload=True)
