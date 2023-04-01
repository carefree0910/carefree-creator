from cfcreator import *
from cfcreator.sdks import *
from cflearn.parameters import OPT


async def main(host: str):
    with OPT.opt_context(dict(host=host)):
        model = Txt2ImgSDModel(text="A lovely little cat.", num_steps=5)
        img = await txt2img(model)
        img.save("txt2img.png")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main("http://localhost:8123"))
