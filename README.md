![noli-creator](./static/images/social-image.jpg)


[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Try%20most%20of%20the%20recent%20AI%20magics%20together,%20online,%20and%20for%20FREE!&url=https://github.com/carefree0910/carefree-creator&via=carefree0910&hashtags=stablediffusion,pytorch,developers)

An open sourced, AI-powered creator for everyone.

- [WebUI](https://creator.nolibox.com/guest) (Recommended!)
- [Google Colab](https://colab.research.google.com/github/carefree0910/carefree-creator/blob/dev/tests/demo.ipynb) (Very limited features, but very customizable!)

> This repo (`carefree-creator`) contains the backend server's codes, the **WebUI** codes (`noli-creator`) will be open sourced as well if it gains enough interests 😉.


### Table of Content

- [tl;dr](#tldr)
- [WebUI & Local Deployment](#webui--local-deployment)
- [Image Generating Features](#image-generating-features)
  - [Text to Image](#text-to-image)
  - [Generate Variations](#generate-variations)
  - [Sketch to Image](#sketch-to-image)
    - [One more thing](#one-more-thing)
    - [General Image to Image translation](#general-image-to-image-translation)
  - [Generate Circular (Tiling) Textures](#generate-circular-tiling-textures)
  - [Generate Better Anime images](#generate-better-anime-images)
  - [Inspect / Copy / Import Parameters](#inspect--copy--import-parameters)
  - [Presets](#presets)
    - [Use Preset Capsules](#use-preset-capsules)
    - [Use Preset Panel](#use-preset-panel)
  - [Outpainting (Experimental)](#outpainting-experimental)
  - [Landscape Synthesis (Experimental)](#landscape-synthesis-experimental)
    - [What determines the size of the generated image?](#what-determines-the-size-of-the-generated-image)
    - [I see many 'holes' in your example, do they matter?](#i-see-many-holes-in-your-example-do-they-matter)
- [Image Processing Features](#image-processing-features)
  - [Super Resolution](#super-resolution)
  - [Inpainting](#inpainting)
- [Installation](#installation)
  - [pip installation](#pip-installation)
    - [Run](#run)
  - [Docker](#docker)
    - [Prepare](#prepare)
    - [Build](#build)
    - [Run](#run-1)
- [Q&A](#qa)
    - [Where are my creations stored?](#where-are-my-creations-stored)
    - [How do I save / load my project?](#how-do-i-save--load-my-project)
    - [How can I contribute to `carefree-creator`?](#how-can-i-contribute-to-carefree-creator)
    - [How can I get my own models interactable on the **WebUI**?](#how-can-i-get-my-own-models-interactable-on-the-webui)
    - [Why no `GFPGAN`?](#why-no-gfpgan)
    - [Is it FREE?](#is-it-free)
    - [Do you like cats?](#do-you-like-cats)
    - [What about dogs?](#what-about-dogs)
    - [Why did you build this project?](#why-did-you-build-this-project)
    - [Will there be a Discord Community?](#will-there-be-a-discord-community)
    - [What is `Nolibox`???](#what-is-nolibox)
- [Known Issues](#known-issues)
- [TODO](#todo)
- [Credits](#credits)

# tl;dr
- An **infinite draw board** for you to save, review and edit all your creations.
- Almost EVERY feature about Stable Diffusion (txt2img, img2img, sketch2img, **variations**, outpainting, circular/tiling textures, sharing, ...).
- Many useful image editing methods (**super resolution**, inpainting, ...).
- Integrations of different Stable Diffusion versions (waifu diffusion, ...).
- GPU RAM optimizations, which makes it possible to enjoy these features with an **NVIDIA GeForce GTX 1080 Ti**!

It might be fair to consider this as:
- An AI-powered, open sourced(*) **Figma**.
- A more 'interactable' **Hugging Face Space**.
- A place where you can try all the exciting and cutting-edge models, together.

> *: The **WebUI** codes are not open sourced **yet**, but we are happy to open source them if it is truely helpful 😉.


# WebUI & Local Deployment

> [Here](#installation) is the deployment guide.

Since `carefree-creator` is a (fairly) stand-alone FastAPI service, it is possible to use our hosted **WebUI** along with your local server. In fact, we've already provided a switch for you:

![use-local-server](./static/images/use-local-server.png)

> The left-most, hand drawing cat is my creation, and `carefree-creator` helped me 'beautify' it a little bit on the right 🤣.
> 
> We will show you how to perform `sketch2img` in [this section](#sketch-to-image).

To make things fancy we can call it a 'Decentralized Deployment Method' (🤨). Anyway, with local deployment, you can then utilize your own machines to avoid waiting my poor cloud server to generate the images for one or few minutes. What's more, since you deployed for yourself, it will be FREE forever!

> This also reveals the goal of `carefree-creator`: we handle the messy **WebUI** parts for you, so you can focus on developing cool models and algorithms that can later seamlessly integrate into it.
> 
> And, with the possibility to deploy locally, you don't have to wait for me to update my poor cloud server. You can simply make a pull request to the `carefree-creator` and tell me: hey, get this feature to the **WebUI** 😆. And after I updated the **WebUI**, you can already play with it on your local machines!
> 
> And of course, as mentioned before, if it gains enough interests, we are happy to open souce the **WebUI** codes as well. In this case, you will have the ability to complete the whole cycle: you can develop your own models, wrap them around to expose APIs, modify **WebUI** to interact with these APIs, and have fun! You can keep your own forks if you want to make them private, or you can make pull requests to the main fork so everyone in the world can also enjoy your works!


# Image Generating Features

Image generating features really opens a brand new world for someone who wants to create but lack of corresponding skills (just like me 🤣). However, generating one single (or, a couple) image at a time without the ability to review/further edit them easily makes creation harder than expected. That's why we support putting all generated images on one single **infinite draw board**, and support trying almost every cool image generating features, together.

The features listed in this section hide behind that picture-icon on the left:

![image-generating-icon](./static/images/image-generating-icon.png)

## Text to Image

This is the most basic and foundamental feature:

![Text to Image](https://github.com/carefree0910/datasets/releases/download/static/text_to_image.gif)

But we added something more. For example, you can choose the style:

![Text to Image With Style](https://github.com/carefree0910/datasets/releases/download/static/text_to_image_with_style.gif)

And as you can see, there are some other options as well - we will cover most of them in the following sections.

## Generate Variations

<details>
<summary>GIF</summary>
<img src="https://github.com/carefree0910/datasets/releases/download/static/variation_generation.gif" alt="Variation Generation" />
</details>

A very powerful feature that we support is to generate variations. Let's say you generated a nice portrait of `komeiji koishi`:

![komeiji koishi](./static/images/koishi0.jpg)

As I've already highlighted, there is a panel called **Variation Generation**. You can simply click the `Generate` button in it and see what happens:

![komeiji koishi](./static/images/koishi1.jpg)

Another `komeiji koishi` appears!

You might have noticed that you can adjust the `Fidelity` of the variation, it indicates how 'similar' the generated image will be to the original image. By lowering it a little bit, you can get even more interesting results:

![komeiji koishi](./static/images/koishi2.jpg)

Cool!

And why not generate variations based on the generated variations:

![komeiji koishi](./static/images/koishi3.jpg)

The last `komeiji koishi` somehow mimics the art style of `ZUN` 😆!

## Sketch to Image

<details>
<summary>GIF</summary>
<img src="https://github.com/carefree0910/datasets/releases/download/static/image_translation.gif" alt="Variation Generation" />
</details>

We support 'translating' any sketches to images with the given prompt. Although it is not required, we recommend adding an 'Empty Node' (with the 'plus' icon on the top) as a 'canvas' for you to draw on:

![add-empty-node](./static/images/sketch0.png)

> You might notice that there is an `Outpainting` panel on the left when you select an Empty Node. We will cover its usage in [this section](#outpainting-experimental).

After our 'canvas' is ready, you can trigger the 'brush' and start drawing!

![drawing](./static/images/sketch1.png)

> The position doesn't really matter, we will always center your sketch before uploading it to our server 😉.

Once you are satisfied with your wonderful sketch, click the `Finish` button on the right, your drawing will then turn into a selectable Node, and an `Image Translation` panel will appear on the left:

![image-translation](./static/images/sketch2.png)

> As you can see, the preview sketch does not contain the 'canvas', that's why we said the 'canvas' is not required.
> 
> When the sketch is uploaded to our server, we will fill the background with white color - so don't use white color to draw 😆!

After inputing some related texts, you can scroll down the `Image Translation` panel and click the `Translate` button:

![image-translation-submit](./static/images/sketch3.png)

And the result should be poped up in a few seconds:

![image-translation-result](./static/images/sketch4.jpg)

Not bad!

### One more thing

You don't actually need to worry whether your drawings could be recognized or not - it turns out that Stable Diffusion is pretty capable of recognizing them 😆:

![image-translation-wild](./static/images/sketch5.jpg)

### General Image to Image translation

Although I'm using a built-in sketch-to-image to illustrate the concepts, the `Image Translation` is in fact a general `img2img` technique, so you can actually apply it to any images. For instance, you can apply it to the generated image:

![image-translation-general](./static/images/sketch6.jpg)

Seems that more details are added!

With this technique, you can actually upload your own images (for instance, the paintings that are drawn by kids), and turn them into an 'art piece':

![image-translation-general](./static/images/img2img.jpg)

## Generate Circular (Tiling) Textures

<details>
<summary>GIF</summary>
<img src="https://github.com/carefree0910/datasets/releases/download/static/circular_textures.gif" alt="Variation Generation" />
</details>

So what are circular textures? Circular textures are images that can be 'tiled' together, and it is easy to specify `carefree-creator` to generate such textures by toggling the corresponding switch:

![circular-textures](./static/images/circular-textures0.jpg)

Hmm, nothing special, right? That's because the magic only happens if you 'tile' them together:

![circular-textures-tile](./static/images/circular-textures1.jpg)

## Generate Better Anime images

Thanks to [Waifu Diffusion](https://github.com/harubaru/waifu-diffusion), we are able to generate better anime images by toggling the corresponding switch:

![waifu-diffusion](./static/images/waifu-diffusion.jpg)

## Inspect / Copy / Import Parameters

It's well known that x-Diffusion models need good 'prompts' to generate good images, but what makes good 'prompts' remains mystery. Therefore, we support inspecting parameters of every generated image:

![inspect-parameters](./static/images/inspect-parameters.jpg)

You can copy the parameters with the little `Copy` button, and the copied parameters can then be pasted to the `Parameters to Image` panel on the left:

![parameters-to-image](./static/images/parameters-to-image.jpg)

In this way, all the creations will be sharable, reproducible and (sort of) understandable!

## Presets

If you want to generate some really fancy images (like the ones that fly around the internet these days), a good starting point is to use our presets.

> And by leveraging the `Inspect Parameters` function metioned in the previous section, we can understand what prompts / parameters are used behind these results, and possibly 'learn' how to master these models!

### Use Preset Capsules

If you scroll down the `Text to Image` panel, you will see a `Try these out!` section with many 'capsules':

![preset-capsules](./static/images/preset-capsules.jpg)

We will generate the corresponding images if you click one of these capsules.

### Use Preset Panel

We also provide a Preset Panel on the left (that nice, little, Pokémon-ish icon 🤣):

![preset-panel](./static/images/preset-panel.jpeg)

Currently we only support Generate Cats 🐱, but we will add more in the future (for instance, Generate Dogs 🐶)!

## Outpainting (Experimental)

We in fact support outpainting algorithm, but I shall be honest: that the Stable Diffusion model is not as good as the DALLE·2 model in this case. So I will simply put a single-image demonstration here:

![outpainting](./static/images/outpainting0.jpeg)

- **0** - Create an Empty Node and drag it to the area that you want to outpaint on
  - It needs to be placed 'below' the original image. The keyboard shortcut is `ctrl+[` for Windows and `cmd+[` for Mac.
- **1** - Expand the `Outpainting` on the left and:
  - Input some texts in the text area.
  - Click the `Mark as Outpainting Area` button.
    - A nice little preview image should then pop up above the text area with this action.
- **2** - Click the `Outpaint` button and wait for the result.

It is likely that some goofy results will appear 🤣. In this case, you can undo it by `ctrl+z` / `cmd+z` and try it one more time. (Maybe) Eventually, you will get nice result.

But - there are some tricks here. If you are trying to outpaint a generated image, recall that you can [copy the parameters](#inspect--copy--import-parameters) of every generated image, so why not use exactly the same prompt to outpaint:

![outpainting-with-same-prompt](./static/images/outpainting1.jpg)

> That's a REALLY long prompt 😆!

And after a few tries, I get this result:

![outpainting-result](./static/images/outpainting2.jpg)

Still far from good, but it's quite interesting!

## Landscape Synthesis (Experimental)

<details>
<summary>GIF</summary>
<img src="https://github.com/carefree0910/datasets/releases/download/static/landscape.gif" alt="Variation Generation" />
</details>

Another interesting feature is that you can do landscape synthesis, similar to `GauGAN`:

![landscape-synthesis-result](./static/images/landscape-synthesis0.jpg)

But again, the result is quite unpredictable, so I will simply put a single-image demonstration here:

![landscape-synthesis](./static/images/landscape-synthesis1.png)

- **0** - Click the landscape icon on the toolbar, and you will enter the 'Landscape drawing' mode.
- **1** - You will draw an area of the landscape per mouse down & mouse up. Before that, you can choose which type of landscape that you are going to draw on the right panel.
- **2** - You can draw wherever you want on the draw board, but better keep everything together.
- **3** - Once you are satisfied with your wonderful sketch, click the `Finish` button on the right, your drawing will then turn into a selectable Node, and a `Landscape Synthesis` button will appear on the right:

![landscape-synthesis-submit](./static/images/landscape-synthesis2.png)

Click it, and the result should be poped up in a few seconds:

![landscape-synthesis-submit](./static/images/landscape-synthesis3.jpg)

Far from good, but not so bad!

### What determines the size of the generated image?

The generated image will have the same size as the sketch, so it will be dangerous if you accidentally submit a HUGE sketch without even noticing:

![landscape-synthesis-dangerous](./static/images/landscape-synthesis4.png)

The sketch looks small, but the actual size is `6765.1 x 4501.5`!! This happened because we support global scaling, and some huge stuffs will 'look small' on the draw board.

### I see many 'holes' in your example, do they matter?

I've implemented something like 'nearest search' to fill those holes, so don't worry: they should be working as expected in most cases!


# Image Processing Features

Apart from the image generating features, we also provided some rather stand-alone image processing features that can be used on any images. Our goal here is to provide an AI-powered toolbox that can do something difficult with only one or a few clicks.

The features listed in this section hide behind that magic-wand-icon on the left:

![image-processing-icon](./static/images/image-processing-icon.jpg)

## Super Resolution

Worried that the generated image is not high-res enough? Then our Super Resolution feature can come to rescue:

![super-resolution](./static/images/super-resolution0.png)

There are two buttons: `Super Resolution` and `Super Resolution (Anime)`. They are basically two versions from `Real ESRGAN`, where the former is a 'general' SR solution, and the latter does some optimizations on anime pictures.

By clicking one of these buttons, you will get a high-res image in a few seconds:

![super-resolution](./static/images/super-resolution1.png)

As you can see, the result even looks like a vector graphic, nice!

> Although you can SR the already SR-ed image, the image size will grow exponentially (`4x` each), and soon explode my (or your, if you deployed locally) machine 😮!

## Inpainting

<details>
<summary>GIF</summary>
<img src="https://github.com/carefree0910/datasets/releases/download/static/inpainting.gif" alt="Variation Generation" />
</details>

Annoyed that only a small part of a generated image is not what you want? Then our Inpainting feature can come to rescue. Let's say we've generated a nice portrait of `hakurei reimu`, but you might notice that there is something weird:

![inpainting-initial](./static/images/inpainting0.jpg)

So let's use our `brush` tool to 'overwrite' the weird area:

![inpainting-brush](./static/images/inpainting1.jpg)

- **0** - Click the brush icon on the toolbar, and you will enter the 'brushing' mode.
- **1** - Trigger the `Use Fill` mode on the right, so it will be convenient to draw areas.
- **2** - Draw the contour of the target area, and the `Use Fill` mode will help you fill the center.

> The color could be any color, not necessary to be green 😉.

After clicking the `Finish` button on the right, the drawing will then turn into a selectable Node, and the `Inpainting` panel on the left can now be utilized:

![inpainting-brush](./static/images/inpainting2.jpg)

1. click the `Mark as Inpainting Mask` to mark your drawing as mask.
2. click the portrait, then click the `Mark as Image` to mark the portrait as background image.

Then the `Inpaint` button should be available, click it and wait for the result:

![inpainting-submit](./static/images/inpainting3.jpg)

Not bad! But can we do something more?

...Yes! We can apply the `Super Resolution (Anime)` on the inpainted image. And here's the final result:

![inpainting-final](./static/images/inpainting4.jpg)

Not perfect, but I'm pretty satisfied because what I've done is just some simple clicking 😆.


# Installation

`carefree-creator` is built on top of `carefree-learn`, and requires:
- Python 3.8 / 3.9
- `pytorch>=1.9.0`. Please refer to [PyTorch](https://pytorch.org/get-started/locally/)'s official website, and it is highly recommended to pre-install PyTorch with conda.

## pip installation

```bash
pip install carefree-creator
```

### Run

```bash
uvicorn apis.interface:app --host 0.0.0.0 --port 8123
```

## Docker

### Prepare

```bash
export TAG_NAME=cfcreator
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
docker run --gpus all --rm -p 8123:8123 -v /full/path/to/your/client/logs:/workplace/apis/logs $TAG_NAME:latest
```


# Q&A

### Where are my creations stored?

They are currently stored on my poor cloud server, and I'm planning to support storing them on your local machines!

### How do I save / load my project?

We will perform an auto-save everytime you make some modifications, and will perform a period saving every minute, to the `localStorage` of your browser. However, I have to admit that they are not as reliable as it should be, so you can download the whole project to your own machines:

![download-project](./static/images/download-project.jpg)

This will download a `.noli` file, which contains all the information you need to fully reconstruct the current draw board. You can then import these `.noli` files later with the `Import Project` menu option (right above the `Download Project` option).

### How can I contribute to `carefree-creator`?

`carefree-creator` is a FastAPI-based service, and I've already made some abstractions so it should be fairly easy to implement a new `Algorithm`.

The development guide is on our [TODO](#todo) list, but here are some brief introductions that might help:

0. the `cfcreator/txt2img.py` file is a good reference.
1. create a new file under the `cfcreator` directory, and in this file:
   1. define the endpoint of your service.
   2. `register` an `Algorithm`, which should contain an `initialize` method and a `run` method.
2. go to `cfcreator/__init__.py` file and import your newly implemented modules here.

### How can I get my own models interactable on the **WebUI**?

As long as we open sourced the **WebUI** you can implement your own UIs, but for now you can contribute to this `carefree-creator` repo and then ask me to do the UI jobs for you (yes, you can be my boss 😆).

### Why no `GFPGAN`?

That's because I think generating real human faces might not be a good practice for `carefree-creator`, so currently I'm not going to develop tool chains around it. If you encountered some scenarios that truly need it, feel free to contact me and let me know!

### Is it FREE?

It will **ALWAYS** be **FREE** if:
- You are using [local deployment](#webui--local-deployment) (Recommended!).
- You are using my own poor cloud server.

For the second situation, if more and more people are using this project, you might be waiting longer and longer. You can inspect where the positions of your tasks are in the waiting queue here:

![pending-panel](./static/images/pending-panel.png)

The number after `pending` will be the position. If it is ridiculously large... Then you may try [local deployment](#webui--local-deployment), or some business will go on (accounts, charges for dedicated cloud servers, etc) 🤣.

> As long as this project is not as famous as those crazy websites, even my poor cloud server should be able to handle the requests, so you can consider it to be FREE in most cases (Not to mention you can always use [local deployment](#webui--local-deployment)) 😉.

### Do you like cats?

I **LOVE** cats. They are soooooo **CUTE**.

### What about dogs?

Dogs are cute as well, but I got bitten when I was young so...

### Why did you build this project?

I've been a big fan of Touhou since 10 years ago, and one of my biggest dreams is to make an epic Touhou fan game.

It wouldn't be possible because I can hardly draw anything (🤣), but now with Stable Diffusion everything is hopeful again.

So the initial reason of building this project is simple: I want to provide a tool that can empower anyone, who is suffering from acquiring game materials, the ability to create ones on their own. That's why we put pretty much attention on the [Variation Generation](#generate-variations) feature, since this is very important for creating a vivid character.

> Stable Diffusion gives me some confidence, and Waifu Diffusion further convinced my determination. Many thanks to these great open source prjects!!!

And as the development goes on, I figure out that this tool has more potential: It could be the '**Operation System**' of the AI generation world! The **models/algorithms** serve as the `softwares`, and your **creations** serve as the `files`. You can always **review/edit** your `files` with the `softwares`, as well as **sharing/importing** them.

In the future, the `softwares` should be easy to **implement/publish/install/uninstall**, and the `files` should be able to store at **cloud/local machine** (currently they are all on cloud, or, on my poor cloud server 🤣).

This will further break the wall between the academic world and the non-academic world. The Hugging Face Space is doing a good job now, but there are still three pain points:

- Its interaction is usable, but still quite restricted.
- The results are generated one after another, we cannot review/edit the results that are generated 5 minutes ago.
- The service is deployed at their own servers, so you have to wait if their servers are busy / not GPU accelerated.

And now, with the ability to do **local deployment**, along with the fantastic **infinite draw board** as the **WebUI**, these pain points will all be solved. Not to mention with some inference technique (such as the `ZeRO` from `deepspeed`), it is possible to deploy huge, huge models even on your laptop, so don't worry about the capability of this system - everything will be possible!

### Will there be a Discord Community?

Unfortunately I'm not familiar with Discord, so if someone can help me build it I will be really appreciated!

### What is `Nolibox`???

`Nolibox` is a startup company where I'm currently working for. Although I have to put the logo everywhere, this project is rather independent and will not be restricted 😉.


# Known Issues

- Undo / Redo in the header toolbar will be messed up when it comes to the 'brushing' mode and 'landscape' mode.
- If you opened two or more tabs of this `creator`, your savings will be messed up because your data is not saved in the cloud, but in the `localStorage` of your browser.
- If you delete an inpainting mask and then undo the deletion, you cannot see the preview image of the inpainting mask anymore until you set another Node as inpainting mask and then switch it back.


# TODO

- [ ] User Guide
- [ ] Textual Inversion
- [ ] Better Outpainting Techniques
- [ ] Development Guide
- [ ] And much more...


# Credits

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion), the foundation of various generation methods.
- [Waifu Diffusion](https://github.com/harubaru/waifu-diffusion), the anime-finetuned version of Stable Diffusion.
- [Real ESRGAN](https://github.com/xinntao/Real-ESRGAN), the adopted Super Resolution methods.
- [Latent Diffusion](https://github.com/CompVis/latent-diffusion), the adopted Inpainting & Landscape Synthesis method.
- [carefree-learn](https://github.com/carefree0910/carefree-learn), the code base that has re-implemented all the models above and provided clean and handy APIs.
- And You! Thank you for watching!
