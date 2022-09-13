Template for utilizing `carefree-client`.


## Modify Notes

- Modify the `PACKAGE_NAME` in `setup.py`.
- Create a folder called `PACKAGE_NAME` at the root dir, this will be your package root.
- Implement your algorithms wherever you want in your package. Just don't forget to register them with `AlgorithmBase`:

```python
from cfclient.models.core import AlgorithmBase

@AlgorithmBase.register("foo")
class Foo(AlgorithmBase):
    ...
```

- Expose APIs of your algorithms in `apis/interface.py` with the help of `get_responses`, `run_algorithm` & `loaded_algorithms`:

```python
endpoint = ...
DataModel = ...
ResponseModel = ...

@app.post(endpoint, responses=get_responses(ResponseModel))
async def hello(data: DataModel) -> ResponseModel:
    # Notice that the key here, `foo`, is identical with the registered name shown above 
    return await run_algorithm(loaded_algorithms["foo"], data)
```

> It is recommended to put the APIs near the `demo` section (L129)


## Prepare

```bash
export TAG_NAME=xxx
```

## Build

```bash
docker build -t $TAG_NAME .
```

If your internet environment lands in China, it might be faster to build with `Dockerfile.cn`:

```bash
docker build -t $TAG_NAME -f Dockerfile.cn .
```


## Run

```bash
docker run --rm -p 8123:8123 -v /full/path/to/your/client/logs:/workplace/apis/logs $TAG_NAME:latest
```

or

```bash
docker run --rm --link image_name_of_your_triton_server -p 8123:8123 -v /full/path/to/your/client/logs:/workplace/apis/logs $TAG_NAME:latest
```

In this case, you need to modify the `apis/interface.py` file as well: you need to modify the `constants` variable (defined at L27) and set the value of `triton_host` (defined at L28) from `None` to `image_name_of_your_triton_server`.
