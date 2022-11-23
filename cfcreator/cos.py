import io
import os
import json
import time
import uuid
import logging
import requests

import numpy as np

from io import BytesIO
from PIL import Image
from enum import Enum
from typing import Union
from typing import BinaryIO
from typing import Optional
from aiohttp import ClientSession
from pydantic import Field
from pydantic import BaseModel
from cfclient.utils import download_image_with_retry as download

from .parameters import use_cos
from .parameters import bypass_audit

try:
    from redis import Redis
    from qcloud_cos import CosConfig
    from qcloud_cos import CosS3Client
except:
    Redis = None
    CosConfig = None
    CosS3Client = None


SCHEME = "http"
REGION = "ap-shanghai"
BUCKET = "ailab-1310750649"
CDN_HOST = "https://ailabcdn.nolibox.com"
COS_HOST = "https://ailab-1310750649.cos.ap-shanghai.myqcloud.com"
TEXT_BIZ_TYPE = "721ba17a279d5074cccd53587610dec4"
IMAGE_BIZ_TYPE = "f64516c2fdf256f9a5d0c45cd4e756a0"
SECRET_ID = os.getenv("SECRETID")
SECRET_KEY = os.getenv("SECRETKEY")

TEMP_TEXT_FOLDER = "tmp_txt"
TEMP_IMAGE_FOLDER = "tmp"
RETRY = 3
PART_SIZE = 20
MAX_THREAD = 1
UPLOAD_RETRY = 1
UPLOAD_TIMEOUT = 30

logger = logging.getLogger(__name__)


class UploadTextResponse(BaseModel):
    path: str = Field(..., description="The path on the cloud.")
    cdn: str = Field(..., description="The `cdn` url of the input text.")
    cos: str = Field(
        ...,
        description="The `cos` url of the input text, which should be used internally.",
    )


class AuditResponse(BaseModel):
    safe: bool = Field(..., description="Whether the input content is safe.")
    reason: str = Field(..., description="If not safe, what's the reason?")


class UploadImageResponse(BaseModel):
    path: str = Field(..., description="The path on the cloud.")
    cdn: str = Field(..., description="The `cdn` url of the input image.")
    cos: str = Field(
        ...,
        description="The `cos` url of the input image, which should be used internally.",
    )


def upload_text(
    client: CosS3Client,
    text: str,
    *,
    folder: str,
    part_size: int = 10,
    max_thread: int = 10,
) -> UploadTextResponse:
    path = f"{folder}/{uuid.uuid4().hex}.txt"
    text_io = io.StringIO(text)
    client.upload_file_from_buffer(
        BUCKET,
        path,
        text_io,
        PartSize=part_size,
        MAXThread=max_thread,
    )
    return UploadTextResponse(
        path=path,
        cdn=f"{CDN_HOST}/{path}",
        cos=f"{COS_HOST}/{path}",
    )


def upload_temp_text(
    client: CosS3Client,
    text: str,
    *,
    part_size: int = 10,
    max_thread: int = 10,
) -> UploadTextResponse:
    return upload_text(
        client,
        text,
        folder=TEMP_TEXT_FOLDER,
        part_size=part_size,
        max_thread=max_thread,
    )


def parse_audit_text(res: dict) -> Optional[AuditResponse]:
    detail = res["JobsDetail"]
    if detail["State"] != "Success":
        return
    label = detail["Label"]
    return AuditResponse(safe=label == "Normal", reason=label)


def audit_text(client: CosS3Client, text: str) -> AuditResponse:
    res = client.ci_auditing_text_submit(
        BUCKET,
        "",
        Content=text.encode("utf-8"),
        BizType=TEXT_BIZ_TYPE,
    )
    job_id = res["JobsDetail"]["JobId"]
    parsed = parse_audit_text(res)
    patience = 20
    interval = 100
    for i in range(patience):
        if parsed is not None:
            break
        time.sleep(interval)
        res = client.ci_auditing_text_query(BUCKET, job_id)
        parsed = parse_audit_text(res)
    if parsed is None:
        return AuditResponse(safe=False, reason=f"Timeout ({patience * interval})")
    return parsed


def upload_image(
    client: CosS3Client,
    inp: Union[bytes, np.ndarray, BinaryIO],
    *,
    folder: str,
    retry: int = UPLOAD_RETRY,
    timeout: int = UPLOAD_TIMEOUT,
    # part_size: int = PART_SIZE,
    # max_thread: int = MAX_THREAD,
) -> UploadImageResponse:
    path = f"{folder}/{uuid.uuid4().hex}.png"
    if isinstance(inp, bytes):
        img_bytes = io.BytesIO(inp)
    elif isinstance(inp, np.ndarray):
        img_bytes = io.BytesIO()
        Image.fromarray(inp).save(img_bytes, "PNG")
        img_bytes.seek(0)
    else:
        img_bytes = inp

    cdn_url = f"{CDN_HOST}/{path}"
    cos_url = f"{COS_HOST}/{path}"

    original_retry = client._retry
    original_timeout = client._conf._timeout
    client._retry = retry
    client._conf._timeout = timeout
    try:
        client.put_object(
            Key=path,
            Body=img_bytes,
            Bucket=BUCKET,
            StorageClass="STANDARD",
        )
    except Exception:
        try:
            raw_data = requests.get(cos_url).content
            Image.open(BytesIO(raw_data)).verify()
            logger.info("\n\ntried to get url after exception and succeeded!!!\n\n")
        except Exception:
            logger.exception("\n\ntried to get url after exception but failed!!!\n\n")
            raise
    finally:
        client._retry = original_retry
        client._conf._timeout = original_timeout
    return UploadImageResponse(
        path=path,
        cdn=cdn_url,
        cos=cos_url,
    )


def upload_temp_image(
    client: CosS3Client,
    inp: Union[bytes, np.ndarray, BinaryIO],
    # *,
    # part_size: int = PART_SIZE,
    # max_thread: int = MAX_THREAD,
) -> UploadImageResponse:
    return upload_image(
        client,
        inp,
        folder=TEMP_IMAGE_FOLDER,
        # part_size=part_size,
        # max_thread=max_thread,
    )


class ForbidEnum(int, Enum):
    NONE = 0
    FROZEN = 1
    MIGRATED = 2


class AuditJobsDetailModel(BaseModel):
    Object: str
    Label: str
    Category: str
    SubLabel: str
    ForbidState: ForbidEnum


def audit_image(audit_client: Redis, path: str, timeout: int = 3) -> AuditResponse:
    if bypass_audit():
        return AuditResponse(safe=True, reason="")
    t = time.time()
    while time.time() - t <= timeout:
        res = audit_client.get(path)
        if res is None:
            time.sleep(0.1)
            continue
        data = AuditJobsDetailModel(**json.loads(res))
        if data.Label == "Normal" and data.ForbidState == ForbidEnum.NONE:
            return AuditResponse(safe=True, reason="")
        reason = data.Label
        if data.Category:
            reason = f"{reason}/{data.Category}"
        if data.SubLabel:
            reason = f"{reason}/{data.SubLabel}"
        return AuditResponse(safe=False, reason=reason)
    return AuditResponse(safe=False, reason="timeout")


async def download_image_with_retry(
    session: ClientSession,
    url: str,
    *,
    retry: int = RETRY,
    interval: int = 1,
) -> Image.Image:
    if use_cos() and url.startswith(CDN_HOST):
        url = url.replace(CDN_HOST, COS_HOST)
    return await download(session, url, retry=retry, interval=interval)


__all__ = [
    "SCHEME",
    "REGION",
    "SECRET_ID",
    "SECRET_KEY",
    "upload_text",
    "upload_temp_text",
    "audit_text",
    "upload_image",
    "upload_temp_image",
    "audit_image",
    "ForbidEnum",
    "AuditJobsDetailModel",
    "AuditResponse",
    "UploadImageResponse",
]


if __name__ == "__main__":
    import redis
    from cfcreator.parameters import audit_redis_kwargs

    config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY)
    cos_client = CosS3Client(config)
    redis_client = redis.Redis(**audit_redis_kwargs())
    print(">>> start uploading")
    # res = upload_temp_image(cos_client, np.zeros([64, 64], np.uint8))
    res = upload_temp_image(cos_client, np.array(Image.open("test.jpg")))
    print(res)
    print(">>> audit image")
    print(audit_image(redis_client, res.path))
    print(">>> upload completed")
