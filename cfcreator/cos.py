import io
import os
import time
import uuid
import requests

import numpy as np

from io import BytesIO
from PIL import Image
from typing import Union
from typing import BinaryIO
from typing import Optional
from aiohttp import ClientSession
from pydantic import Field
from pydantic import BaseModel
from cfclient.utils import download_image_with_retry as download

from .parameters import use_cos

try:
    from qcloud_cos import CosConfig
    from qcloud_cos import CosS3Client
except:
    CosConfig = None
    CosS3Client = None


SCHEME = "http"
REGION = "ap-shanghai"
BUCKET = "ailab-1310750649"
CDN_HOST = "https://ailabcdn.nolibox.com"
COS_HOST = "https://ailab-1310750649.cos.ap-shanghai.myqcloud.com"
TEXT_BIZ_TYPE = "56daee337ae2d847e55838c0ddb6d547"
IMAGE_BIZ_TYPE = "aeb9f1736399386b4bdcc5496ece5289"
SECRET_ID = os.getenv("SECRETID")
SECRET_KEY = os.getenv("SECRETKEY")

TEMP_TEXT_FOLDER = "tmp_txt"
TEMP_IMAGE_FOLDER = "tmp"
RETRY = 3
PART_SIZE = 20
MAX_THREAD = 1
UPLOAD_RETRY = 1
UPLOAD_TIMEOUT = 30


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
    part_size: int = PART_SIZE,
    max_thread: int = MAX_THREAD,
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
        client.upload_file_from_buffer(
            BUCKET,
            path,
            img_bytes,
            PartSize=part_size,
            MAXThread=max_thread,
        )
    except Exception:
        try:
            raw_data = requests.get(cos_url).content
            Image.open(BytesIO(raw_data)).verify()
        except Exception:
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
    *,
    part_size: int = PART_SIZE,
    max_thread: int = MAX_THREAD,
) -> UploadImageResponse:
    return upload_image(
        client,
        inp,
        folder=TEMP_IMAGE_FOLDER,
        part_size=part_size,
        max_thread=max_thread,
    )


def audit_image(client: CosS3Client, path: str) -> AuditResponse:
    res = client.get_object_sensitive_content_recognition(
        BUCKET,
        path,
        BizType=IMAGE_BIZ_TYPE,
    )
    label = res["Label"]
    return AuditResponse(safe=label == "Normal", reason=label)


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
    "AuditResponse",
    "UploadImageResponse",
]


if __name__ == "__main__":
    config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY)
    cos_client = CosS3Client(config)
    print(">>> start uploading")
    print(upload_temp_image(cos_client, np.zeros([64, 64], np.uint8)))
    print(">>> upload completed")
