import os
import json
import requests

from cfcreator import *
from cfcreator.sdks import *
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from cftool.misc import print_info
from cflearn.parameters import OPT


test_parameters = {
    txt2img_sd_endpoint: {"text": "A lovely little cat."},
    txt2img_sd_inpainting_endpoint: {
        "text": "a large tree",
        "url": "https://ailabcdn.nolibox.com/tmp/3cf1b7cc46fc496bb758ed56c20f8195.png",
        "mask_url": "https://ailabcdn.nolibox.com/tmp/31c8575a2fd24a9d9a840266073a0523.png",
    },
    txt2img_sd_outpainting_endpoint: {
        "url": "https://ailab-huawei-cdn.nolibox.com/upload/images/f1b0e606c8df49efa0d6dbf831576997.png",
        "text": "Blue cars are driving on the road, lakes, sunset, clean sky, superb",
    },
    txt2txt_prompt_enhance_endpoint: {"text": "A lovely little cat."},
    img2img_sd_endpoint: {
        "url": "https://ailab-huawei-cdn.nolibox.com/upload/images/100cdc5685c3446eb1d7ce57e70eb682.png",
        "text": "(((masterpiece))),(((best quality))),((ultra-detailed)),((an extremely delicate and beautiful)), (beautiful detailed eyes), 1girl, (undine), (touhou), (((((komeiji_koishi)))))",
        "version": "anime_anything",
    },
    img2img_semantic2img_endpoint: {
        "url": "https://ailab-huawei-cdn.nolibox.com/upload/images/1137f95fed6a44ce85d176131494a407.png",
        "color2label": {
            "rgb(170,170,170)": 106,
            "rgb(29,195,49)": 124,
            "rgb(200,146,117)": 127,
            "rgb(116,77,52)": 135,
            "rgb(157,157,255)": 148,
            "rgb(54,62,167)": 155,
            "rgb(95,219,255)": 157,
            "rgb(230,182,19)": 162,
            "rgb(128,255,255)": 178,
        },
    },
    img2img_sr_endpoint: {
        "url": "https://ailabcdn.nolibox.com/tmp/3cf1b7cc46fc496bb758ed56c20f8195.png",
        "is_anime": True,
        "target_w": 1024,
        "target_h": 1024,
    },
    img2img_inpainting_endpoint: {
        "model": "lama",
        "url": "https://ailab-huawei-cdn.nolibox.com/upload/images/ff9b03566cc6448498c1845ee12fc8b8.png",
        "mask_url": "https://ailab-huawei-cdn.nolibox.com/upload/images/bdf4cf43e4974922917fd5a93c4f335c.png",
    },
    img2img_harmonization_endpoint: {
        "url": "https://ailab-huawei-cdn.nolibox.com/upload/images/ff9b03566cc6448498c1845ee12fc8b8.png",
        "mask_url": "https://ailab-huawei-cdn.nolibox.com/upload/images/bdf4cf43e4974922917fd5a93c4f335c.png",
        "strength": 1.0,
    },
    img2img_sod_endpoint: {
        "url": "https://ailab-huawei-cdn.nolibox.com/upload/images/ff9b03566cc6448498c1845ee12fc8b8.png"
    },
    img2txt_caption_endpoint: {
        "url": "https://ailab-huawei-cdn.nolibox.com/upload/images/ff9b03566cc6448498c1845ee12fc8b8.png"
    },
    control_depth_endpoint: {
        "url": "https://ailab-huawei-cdn.nolibox.com/aigc-private/images/ef470a0e9566412aabdd17284db55675.png",
        "prompt": "a lovely little cat",
        "negative_prompt": "appendix, malformed, disfigured, ugly, deformed, mutilated, morbid, bad anatomy, cropped, blurred, mutated, error, lowres, blurry, low quality, signature, watermark, text",
        "fidelity": 0.05,
        "use_img2img": True,
        "seed": 123,
    },
    control_canny_endpoint: {
        "url": "https://ailab-huawei-cdn.nolibox.com/aigc-private/images/ef470a0e9566412aabdd17284db55675.png",
        "prompt": "a lovely little cat",
        "negative_prompt": "appendix, malformed, disfigured, ugly, deformed, mutilated, morbid, bad anatomy, cropped, blurred, mutated, error, lowres, blurry, low quality, signature, watermark, text",
        "fidelity": 0.05,
        "use_img2img": True,
        "seed": 123,
    },
    control_pose_endpoint: {
        "url": "https://ailab-huawei-cdn.nolibox.com/aigc-private/images/ef470a0e9566412aabdd17284db55675.png",
        "prompt": "a lovely little cat",
        "negative_prompt": "appendix, malformed, disfigured, ugly, deformed, mutilated, morbid, bad anatomy, cropped, blurred, mutated, error, lowres, blurry, low quality, signature, watermark, text",
        "fidelity": 0.05,
        "use_img2img": True,
        "seed": 123,
    },
    control_mlsd_endpoint: {
        "url": "https://ailab-huawei-cdn.nolibox.com/aigc-private/images/ef470a0e9566412aabdd17284db55675.png",
        "prompt": "a lovely little cat",
        "negative_prompt": "appendix, malformed, disfigured, ugly, deformed, mutilated, morbid, bad anatomy, cropped, blurred, mutated, error, lowres, blurry, low quality, signature, watermark, text",
        "fidelity": 0.05,
        "use_img2img": True,
        "seed": 123,
    },
    control_multi_endpoint: {
        "url": "https://ailab-huawei-cdn.nolibox.com/aigc-private/images/ef470a0e9566412aabdd17284db55675.png",
        "types": ["canny", "depth"],
        "prompt": "a lovely little cat",
        "negative_prompt": "appendix, malformed, disfigured, ugly, deformed, mutilated, morbid, bad anatomy, cropped, blurred, mutated, error, lowres, blurry, low quality, signature, watermark, text",
        "fidelity": 0.05,
        "use_img2img": True,
        "seed": 123,
    },
    control_depth_hint_endpoint: {
        "url": "https://ailab-huawei-cdn.nolibox.com/aigc-private/images/ef470a0e9566412aabdd17284db55675.png"
    },
    control_canny_hint_endpoint: {
        "url": "https://ailab-huawei-cdn.nolibox.com/aigc-private/images/ef470a0e9566412aabdd17284db55675.png"
    },
    control_pose_hint_endpoint: {
        "url": "https://ailab-huawei-cdn.nolibox.com/aigc-private/images/ef470a0e9566412aabdd17284db55675.png"
    },
    control_mlsd_hint_endpoint: {
        "url": "https://ailab-huawei-cdn.nolibox.com/aigc-private/images/ef470a0e9566412aabdd17284db55675.png"
    },
}


def download_image(url: str) -> Image.Image:
    return Image.open(BytesIO(requests.get(url).content))


async def main(host: str):
    results = {}
    exceptions = {}
    image_folder = "images"
    with OPT.opt_context(dict(host=host)):
        for endpoint, params in tqdm(list(test_parameters.items())):
            uid = await push(endpoint, params)
            res = await poll(uid)
            task = endpoint2algorithm(endpoint)
            status = res["status"]
            (results if status == Status.FINISHED else exceptions)[task] = res
            if task.startswith("control"):
                task_folder = os.path.join(image_folder, task)
                os.makedirs(task_folder, exist_ok=True)
                response = res["data"]["response"]
                for i, hint_url in enumerate(response["hint_urls"]):
                    i_path = os.path.join(task_folder, f"hint_{i}.png")
                    download_image(hint_url).save(i_path)
                for i, result_url in enumerate(response["result_urls"]):
                    i_path = os.path.join(task_folder, f"result_{i}.png")
                    download_image(result_url).save(i_path)
                continue
            cdn = res.get("data", {}).get("cdn")
            if cdn is not None:
                os.makedirs(image_folder, exist_ok=True)
                image_path = os.path.join(image_folder, f"{task}.png")
                download_image(cdn).save(image_path)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("exceptions.json", "w") as f:
        json.dump(exceptions, f, indent=2)
    print_info(f"num success   : {len(results)}")
    print_info(f"num exception : {len(exceptions)}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main("https://your-kafka-host.com"))
