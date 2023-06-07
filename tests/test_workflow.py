from cfcreator.endpoints import *
from cfcreator.sdks.apis import *


workflow = Workflow()
workflow.push(
    WorkNode(
        key="txt2img_0",
        endpoint=txt2img_sd_endpoint,
        injections={},
        data=dict(text="A lovely little girl.", seed="123", version="anime_anything"),
    )
)
workflow.push(
    WorkNode(
        key="img2img_0",
        endpoint=img2img_sd_endpoint,
        injections=dict(
            txt2img_0=dict(index=0, field="url"),
        ),
        data=dict(
            text="A lovely little girl.",
            seed="345",
            version="anime_anything",
            fidelity=0.6,
        ),
    )
)
workflow.push(
    WorkNode(
        key="controlnet_0",
        endpoint=new_control_multi_endpoint,
        injections=dict(
            txt2img_0=dict(index=0, field="controls.0.data.hint_url"),
            img2img_0=dict(index=0, field="controls.1.data.hint_url"),
        ),
        data=dict(
            prompt="A lovely little girl.",
            seed="567",
            base_model="anime_anything",
            controls=[
                dict(type="depth", data=dict(control_strength=0.8)),
                dict(type="canny", data=dict(control_strength=0.6)),
            ],
        ),
    )
)


if __name__ == "__main__":
    import os
    import asyncio

    apis = APIs(
        focuses_endpoints=[
            txt2img_sd_endpoint,
            img2img_sd_endpoint,
            new_control_multi_endpoint,
        ]
    )
    target = "controlnet_0"
    results = asyncio.run(apis.execute(workflow, target))
    output_folder = "workflow"
    os.makedirs(output_folder, exist_ok=True)
    for i, (k, res) in enumerate(results.items()):
        for j, image in enumerate(res):
            image.save(os.path.join(output_folder, f"{i}_{k}_{j}.png"))
    asyncio.run(apis.destroy())
