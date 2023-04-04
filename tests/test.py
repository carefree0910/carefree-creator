from cfcreator import *
from cfcreator.sdks import *


async def main():
    sdk = DirectSDK()
    model = Txt2ImgSDModel(text="A lovely little cat.", num_steps=5)
    img = await sdk.txt2img(model)
    img.save("txt2img.png")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
