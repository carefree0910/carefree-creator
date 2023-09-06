from typing import List

from .cv import *
from .txt2img import *
from .txt2txt import *
from .img2img import *
from .img2txt import *
from .control import *
from .control_multi import *
from .pipeline import *
from .parameters import Focus
from .legacy.control import *
from .legacy.control_multi import *
from .upscale import *
from .third_party import *
from .workflow import *


endpoint_to_focuses = {
    # txt2img
    txt2img_sd_endpoint: [
        Focus.ALL,
        Focus.SD,
        Focus.SD_BASE,
        Focus.CONTROL,
    ],
    txt2img_sd_inpainting_endpoint: [Focus.ALL, Focus.SD],
    txt2img_sd_outpainting_endpoint: [Focus.ALL, Focus.SD],
    # txt2txt
    txt2txt_prompt_enhance_endpoint: [Focus.ALL, Focus.SYNC],
    # img2img
    img2img_sd_endpoint: [
        Focus.ALL,
        Focus.SD,
        Focus.SD_BASE,
        Focus.CONTROL,
    ],
    img2img_semantic2img_endpoint: [Focus.ALL],
    img2img_sr_endpoint: [Focus.ALL, Focus.SYNC],
    img2img_inpainting_endpoint: [Focus.ALL, Focus.SYNC],
    img2img_harmonization_endpoint: [Focus.ALL, Focus.SYNC],
    img2img_sod_endpoint: [Focus.ALL, Focus.SYNC],
    # img2txt
    img2txt_caption_endpoint: [Focus.ALL, Focus.SYNC],
    # cv
    cv_blur_endpoint: [Focus.ALL, Focus.CV, Focus.SYNC],
    cv_grayscale_endpoint: [Focus.ALL, Focus.CV, Focus.SYNC],
    cv_erode_endpoint: [Focus.ALL, Focus.CV, Focus.SYNC],
    cv_resize_endpoint: [Focus.ALL, Focus.CV, Focus.SYNC],
    cv_affine_endpoint: [Focus.ALL, Focus.CV, Focus.SYNC],
    cv_get_mask_endpoint: [Focus.ALL, Focus.CV, Focus.SYNC],
    cv_inverse_endpoint: [Focus.ALL, Focus.CV, Focus.SYNC],
    cv_fill_bg_endpoint: [Focus.ALL, Focus.CV, Focus.SYNC],
    cv_get_size_endpoint: [Focus.ALL, Focus.CV, Focus.SYNC],
    cv_modify_box_endpoint: [Focus.ALL, Focus.CV, Focus.SYNC],
    cv_generate_masks_endpoint: [Focus.ALL, Focus.CV, Focus.SYNC],
    cv_crop_image_endpoint: [Focus.ALL, Focus.CV, Focus.SYNC],
    cv_histogram_match_endpoint: [Focus.ALL, Focus.CV, Focus.SYNC],
    cv_image_similarity_endpoint: [Focus.ALL, Focus.CV, Focus.SYNC],
    cv_repositioning_endpoint: [Focus.ALL, Focus.CV, Focus.SYNC],
    # ControlNet
    control_depth_endpoint: [Focus.ALL, Focus.CONTROL],
    control_canny_endpoint: [Focus.ALL, Focus.CONTROL],
    control_pose_endpoint: [Focus.ALL, Focus.CONTROL],
    control_mlsd_endpoint: [Focus.ALL, Focus.CONTROL],
    control_multi_endpoint: [Focus.ALL, Focus.CONTROL],
    new_control_depth_endpoint: [Focus.ALL, Focus.CONTROL],
    new_control_canny_endpoint: [Focus.ALL, Focus.CONTROL],
    new_control_pose_endpoint: [Focus.ALL, Focus.CONTROL],
    new_control_mlsd_endpoint: [Focus.ALL, Focus.CONTROL],
    new_control_multi_endpoint: [Focus.ALL, Focus.CONTROL],
    control_depth_hint_endpoint: [Focus.ALL, Focus.SYNC, Focus.CONTROL],
    control_canny_hint_endpoint: [Focus.ALL, Focus.SYNC, Focus.CONTROL],
    control_pose_hint_endpoint: [Focus.ALL, Focus.SYNC, Focus.CONTROL],
    control_mlsd_hint_endpoint: [Focus.ALL, Focus.SYNC, Focus.CONTROL],
    # pipeline
    paste_pipeline_endpoint: [Focus.ALL, Focus.SYNC, Focus.PIPELINE],
    # upscale
    upscale_tile_endpoint: [Focus.ALL, Focus.CONTROL],
    # third party
    facexlib_parse_endpoint: [Focus.ALL, Focus.SYNC],
    facexlib_detect_endpoint: [Focus.ALL, Focus.SYNC],
    # workflow
    workflow_endpoint: [Focus.ALL, Focus.SYNC],
}


def get_endpoints(focus: Focus) -> List[str]:
    return [e for e, focuses in endpoint_to_focuses.items() if focus in focuses]
