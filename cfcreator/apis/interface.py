import os
import json
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


app = FastAPI()
root = os.path.dirname(__file__)

constants = dict(
    triton_host=None,
    triton_port=8000,
    model_root=os.path.join(root, "models"),
    token_root=os.path.join(root, "tokens"),
)

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


@app.post("/translate", responses=get_responses(GetPromptResponse))
@app.post("/get_prompt", responses=get_responses(GetPromptResponse))
def get_prompt(data: GetPromptModel) -> GetPromptResponse:
    return GetPromptResponse(text=data.text, success=True, reason="")


# switch local checkpoint


class ModelRootResponse(BaseModel):
    root: str


class SwitchCheckpointModel(BaseModel):
    key: str
    model: str


class SwitchCheckpointResponse(BaseModel):
    success: bool
    reason: str


class AvailableVersions(BaseModel):
    versions: List[str]


class AvailableModels(BaseModel):
    models: List[str]


class SwitchCheckpointRootModel(BaseModel):
    root: str


class SwitchCheckpointRootResponse(BaseModel):
    success: bool
    reason: str


class ResetCheckpointModel(BaseModel):
    version: str


class ResetCheckpointResponse(BaseModel):
    success: bool
    reason: str


def _get_available_local_models(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return list(
        filter(
            lambda file: file.endswith(".pt") or file.endswith(".ckpt"),
            os.listdir(root),
        )
    )


@app.post("/model_root", responses=get_responses(GetPromptResponse))
def get_model_root() -> ModelRootResponse:
    return ModelRootResponse(root=constants["model_root"])


@app.post("/available_versions", responses=get_responses(GetPromptResponse))
def get_available_api_versions() -> AvailableVersions:
    return AvailableVersions(versions=available_apis())


@app.post("/available_models", responses=get_responses(GetPromptResponse))
def get_available_local_models() -> AvailableModels:
    return AvailableModels(models=_get_available_local_models(constants["model_root"]))


@app.post("/switch", responses=get_responses(SwitchCheckpointResponse))
def switch_checkpoint(data: SwitchCheckpointModel) -> SwitchCheckpointResponse:
    api = get_api(data.key)
    if api is None:
        return SwitchCheckpointResponse(
            success=False,
            reason=f"'{data.key}' is not a valid key, available keys are: {', '.join(available_apis())}",
        )
    model_path = os.path.join(constants["model_root"], data.model)
    if not os.path.isfile(model_path):
        return SwitchCheckpointResponse(
            success=False,
            reason=f"cannot find '{data.model}' under '{constants['model_root']}'",
        )
    try:
        cflearn.scripts.sd.convert(model_path, api, load=True)
        return SwitchCheckpointResponse(success=True, reason="")
    except Exception as err:
        logging.exception(err)
        return SwitchCheckpointResponse(success=False, reason=get_err_msg(err))


@app.post("/reset", responses=get_responses(ResetCheckpointResponse))
def reset_checkpoint(data: ResetCheckpointModel) -> ResetCheckpointResponse:
    err_msg = f"'{data.version}' is not a valid version, available versions are: {', '.join(available_apis())}"
    current = get_api(data.version)
    if current is None:
        return ResetCheckpointResponse(success=False, reason=err_msg)
    init_fn = get_init_fn(data.version)
    if init_fn is None:
        return ResetCheckpointResponse(success=False, reason=err_msg)
    try:
        with init_fn("cpu").load_context() as wrapper:
            od = wrapper.state_dict()
        cflearn.scripts.sd.inject(od, current)
        return ResetCheckpointResponse(success=True, reason="")
    except Exception as err:
        return ResetCheckpointResponse(success=False, reason=get_err_msg(err))


@app.post("/switch_root", responses=get_responses(SwitchCheckpointRootResponse))
def switch_root(data: SwitchCheckpointRootModel) -> SwitchCheckpointRootResponse:
    if not _get_available_local_models(data.root):
        return SwitchCheckpointRootResponse(
            success=False,
            reason=f"cannot find any checkpoints under '{data.root}'",
        )
    constants["model_root"] = data.root
    return SwitchCheckpointRootResponse(success=True, reason="")


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


@app.post("/inject_tokens", responses=get_responses(InjectCustomTokenResponse))
def inject_custom_tokens(data: InjectCustomTokenModel) -> InjectCustomTokenResponse:
    if not _inject_custom_tokens(data.root):
        return InjectCustomTokenResponse(
            success=False,
            reason=f"cannot find any tokens under '{data.root}'",
        )
    return InjectCustomTokenResponse(success=True, reason="")


# meta


env_opt_json = os.environ.get(OPT_ENV_KEY)
if env_opt_json is not None:
    OPT.update(json.loads(env_opt_json))
focus = OPT.get("focus", "all")
registered_algorithms = set()


def register_endpoint(endpoint: str) -> None:
    name = endpoint[1:].replace("/", "_")
    algorithm_name = endpoint2algorithm(endpoint)
    algorithm: IAlgorithm = all_algorithms[algorithm_name]
    registered_algorithms.add(algorithm_name)
    data_model = algorithm.model_class
    if algorithm.response_model_class is None:
        response_model = Response
        post_kwargs = get_image_response_kwargs()
    else:
        response_model = algorithm.response_model_class
        post_kwargs = dict(responses=get_responses(response_model))

    @app.post(endpoint, **post_kwargs, name=name)
    async def _(data: data_model) -> response_model:
        if (
            isinstance(data, (Txt2ImgSDModel, Img2ImgSDModel))
            and not data.custom_embeddings
        ):
            data.custom_embeddings = {
                token: embedding
                for token, embedding in custom_embeddings.items()
                if token in data.text or token in data.negative_prompt
            }
        return await run_algorithm(algorithm, data)


# txt2img
if focus in ("all", "sd", "sd.base", "sd.anime", "control"):
    register_endpoint(txt2img_sd_endpoint)
if focus in ("all", "sd"):
    register_endpoint(txt2img_sd_inpainting_endpoint)
    register_endpoint(txt2img_sd_outpainting_endpoint)

# img2img
if focus in ("all", "sd", "sd.base", "sd.anime", "control"):
    register_endpoint(img2img_sd_endpoint)
if focus == "all":
    register_endpoint(img2img_semantic2img_endpoint)
if focus in ("all", "sync"):
    register_endpoint(img2img_sr_endpoint)
    register_endpoint(img2img_inpainting_endpoint)
    register_endpoint(img2img_harmonization_endpoint)
    register_endpoint(img2img_sod_endpoint)

# img2txt
if focus in ("all", "sync"):
    register_endpoint(img2txt_caption_endpoint)

# txt2txt
if focus in ("all", "sync"):
    register_endpoint(txt2txt_prompt_enhance_endpoint)

# cv
if focus in ("all", "sync"):
    register_endpoint(cv_affine_endpoint)
    register_endpoint(cv_histogram_match_endpoint)

# ControlNet
if focus in ("all", "control"):
    register_endpoint(control_depth_endpoint)
    register_endpoint(control_canny_endpoint)
    register_endpoint(control_pose_endpoint)
    register_endpoint(control_mlsd_endpoint)
    register_endpoint(control_multi_endpoint)
if focus in ("all", "sync", "control"):
    register_endpoint(control_depth_hint_endpoint)
    register_endpoint(control_canny_hint_endpoint)
    register_endpoint(control_pose_hint_endpoint)
    register_endpoint(control_mlsd_hint_endpoint)

# pipeline
if focus in ("all", "sync", "pipeline"):
    register_endpoint(paste_pipeline_endpoint)


# events


@app.on_event("startup")
async def startup() -> None:
    http_client.start()
    OPT["use_cos"] = False
    for k, v in all_algorithms.items():
        if k in registered_algorithms:
            v.initialize()
    _inject_custom_tokens(constants["token_root"])
    print("> Server is Ready!")


@app.on_event("shutdown")
async def shutdown() -> None:
    await http_client.stop()


# schema

app.openapi = carefree_schema


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("interface:app", host="0.0.0.0", port=8989, reload=True)
