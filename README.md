![carefree-learn][socialify-image]


An AI-powered creator for everyone.


# UI

## [Google Colab](https://colab.research.google.com/github/carefree0910/carefree-creator/blob/dev/tests/demo.ipynb)


# Server


## pip

### Install

```bash
pip install -e .
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
docker run --rm -p 8123:8123 -v /full/path/to/your/client/logs:/workplace/apis/logs $TAG_NAME:latest
```

or

```bash
docker run --rm --link image_name_of_your_triton_server -p 8123:8123 -v /full/path/to/your/client/logs:/workplace/apis/logs $TAG_NAME:latest
```

In this case, you need to modify the `apis/interface.py` file as well: you need to modify the `constants` variable (defined at L27) and set the value of `triton_host` (defined at L28) from `None` to `image_name_of_your_triton_server`.


[socialify-image]: https://socialify.git.ci/carefree0910/carefree-learn/image?description=1&descriptionEditable=Deep%20Learning%20%E2%9D%A4%EF%B8%8F%20PyTorch&forks=1&issues=1&logo=https%3A%2F%2Fraw.githubusercontent.com%2Fcarefree0910%2Fcarefree-learn-doc%2Fmaster%2Fstatic%2Fimg%2Flogo.min.svg&pattern=Floating%20Cogs&stargazers=1