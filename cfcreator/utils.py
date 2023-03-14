import cv2
import math

import numpy as np

from typing import List


def resize_image(input_image: np.ndarray, resolution: int) -> np.ndarray:
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(
        input_image,
        (W, H),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA,
    )
    return img


def to_canvas(results: List[np.ndarray], *, padding: int = 0) -> np.ndarray:
    num_results = len(results)
    num_col = math.ceil(math.sqrt(num_results))
    num_row = round(num_results / num_col)
    if num_row * num_col < num_results:
        num_row += 1
    h, w = results[0].shape[:2]
    canvas_w = num_col * w + (num_col - 1) * padding
    canvas_h = num_row * h + (num_row - 1) * padding
    canvas = np.full([canvas_h, canvas_w, 3], 255, np.uint8)
    for i, out in enumerate(results):
        ih, iw = out.shape[:2]
        if h != ih:
            raise ValueError(f"`h` mismatch: {ih} != {h}")
        if w != iw:
            raise ValueError(f"`w` mismatchh: {iw} != {w}")
        ix = i % num_col
        iy = i // num_col
        ix = ix * w + ix * padding
        iy = iy * h + iy * padding
        canvas[iy : iy + h, ix : ix + w] = out
    return canvas
