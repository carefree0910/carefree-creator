![noli-creator](./static/images/social-image.jpg)

An open sourced, AI-powered creator for everyone.

> * This is the backend project of the `Creator` product. If you are looking for the **WebUI** codes, you may checkout the [`carefree-drawboard`](https://github.com/carefree0910/carefree-drawboard) 🎨 project.
>
> * Most of the contents have been moved to the [Wiki](https://github.com/carefree0910/carefree-creator/wiki) page.

<div align="center">

### [Wiki](https://github.com/carefree0910/carefree-creator/wiki) | [WebUI Codes](https://github.com/carefree0910/carefree-drawboard)

<div align="left">

# Installation

`carefree-creator` is built on top of `carefree-learn`, and requires:
- `Python>=3.8`
- `pytorch>=1.12.0`. Please refer to [PyTorch](https://pytorch.org/get-started/locally/)'s official website, and it is highly recommended to pre-install PyTorch with conda.

## Hardware Requirements

> Related issue: [#10](https://github.com/carefree0910/carefree-creator/issues/10).

This project will eat up 11~13 GB of GPU RAM if no modifications are made, because it actually integrates FIVE different SD versions together, and many other models as well. 🤣

There are two ways that can reduce the usage of GPU RAM - lazy loading and partial loading, see the following [`Run`](#run) section for more details.

## pip installation

```bash
pip install carefree-creator
```

If you are interested in the latest features, you may use `pip` to install from source as well:

```bash
git clone https://github.com/carefree0910/carefree-creator.git
cd carefree-creator
pip install -e .
```

### Run

`carefree-creator` builds a CLI for you to setup your local service. For instance, we can:

```bash
cfcreator serve
```

If you don't have an NVIDIA GPU (e.g. mac), you may try:

```bash
cfcreator serve --cpu
```

If you are using your GPU-powered laptop, you may try:

```bash
cfcreator serve --limit 1
```

> The `--limit` flag is used to limit the number of loading models. By specifying `1`, only the executing model will be loaded, and other models will stay on your disk.
>
> See [#10](https://github.com/carefree0910/carefree-creator/issues/10#issuecomment-1520661893) for more details.

If you have plenty of RAM resources but your GPU RAM is not large enough, you may try:

```bash
cfcreator serve --lazy
```

> With the `--lazy` flag, the models will be loaded to RAM, and only the executing model will be moved to GPU RAM.
> 
> So as an exchange, your RAM will be eaten up! 🤣

If you only want to try the SD basic endpoints, you may use:

```bash
cfcreator serve --focus sd.base
```

And if you only want to try the SD anime endpoints, you may use:

```bash
cfcreator serve --focus sd.anime
```

More usages could be found by:

```bash
cfcreator serve --help
```

## Docker

### Prepare

```bash
export TAG_NAME=cfcreator
git clone https://github.com/carefree0910/carefree-creator.git
cd carefree-creator
```

### Build

```bash
docker build -t $TAG_NAME .
```

If your internet environment lands in China, it might be faster to build with `Dockerfile.cn`:

```bash
docker build -t $TAG_NAME -f Dockerfile.cn .
```

### Run

```bash
docker run --gpus all --rm -p 8123:8123 $TAG_NAME:latest
```

# Credits

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion), the foundation of various generation methods.
- [Stable Diffusion from runwayml](https://github.com/runwayml/stable-diffusion), the adopted SD-inpainting method.
- [Waifu Diffusion](https://github.com/harubaru/waifu-diffusion), the anime-finetuned version of Stable Diffusion.
- [Real ESRGAN](https://github.com/xinntao/Real-ESRGAN), the adopted Super Resolution methods.
- [Latent Diffusion](https://github.com/CompVis/latent-diffusion), the adopted Inpainting & Landscape Synthesis method.
- [carefree-learn](https://github.com/carefree0910/carefree-learn), the code base that has re-implemented all the models above and provided clean and handy APIs.
- And You! Thank you for watching!
