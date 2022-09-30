import io
import os
import uuid

import numpy as np

from PIL import Image
from typing import Union
from typing import BinaryIO
from pydantic import Field
from pydantic import BaseModel
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client


REGION = "ap-shanghai"
BUCKET = "ailab-1310750649"
CDN_HOST = "https://ailabcdn.nolibox.com"
COS_HOST = "https://ailab-1310750649.cos.ap-shanghai.myqcloud.com/"
SECRET_ID = os.getenv("SECRETID")
SECRET_KEY = os.getenv("SECRETKEY")

TEMP_FOLDER = "tmp"


class UploadResponse(BaseModel):
    cdn: str = Field(..., description="The `cdn` url of the input image.")
    cos: str = Field(..., description="The `cos` url of the input image, which should be used internally.")


def upload_image(
    client: CosS3Client,
    inp: Union[bytes, np.ndarray, BinaryIO],
    *,
    folder: str,
    part_size: int = 10,
    max_thread: int = 10,
) -> UploadResponse:
    temp_path = f"{folder}/{uuid.uuid4().hex}.png"
    if isinstance(inp, bytes):
        img_bytes = io.BytesIO(inp)
    elif isinstance(inp, np.ndarray):
        img_bytes = io.BytesIO()
        Image.fromarray(inp).save(img_bytes, "PNG")
        img_bytes.seek(0)
    else:
        img_bytes = inp
    client.upload_file_from_buffer(BUCKET, temp_path, img_bytes, PartSize=part_size, MAXThread=max_thread)
    return UploadResponse(
        cdn=f"{CDN_HOST}/{temp_path}",
        cos=f"{COS_HOST}/{temp_path}",
    )

def upload_temp_image(
    client: CosS3Client,
    inp: Union[bytes, np.ndarray, BinaryIO],
    *,
    part_size: int = 10,
    max_thread: int = 10,
) -> UploadResponse:
    return upload_image(
        client,
        inp,
        folder=TEMP_FOLDER,
        part_size=part_size,
        max_thread=max_thread,
    )


__all__ = [
    "REGION",
    "SECRET_ID",
    "SECRET_KEY",
    "upload_image",
    "upload_temp_image",
    "UploadResponse",
]


if __name__ == "__main__":
    config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY)
    cos_client = CosS3Client(config)
    print(">>> start uploading")
    print(upload_temp_image(cos_client, np.zeros([64, 64], np.uint8)))
    print(">>> upload completed")
