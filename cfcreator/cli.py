import os
import click
import uvicorn

from cfcreator import opt_env_context
from cfcreator import Focus
from cflearn.parameters import OPT


@click.group()
def main() -> None:
    pass


@main.command()
@click.option(
    "-p",
    "--port",
    default=8123,
    show_default=True,
    type=int,
    help="Port of the service.",
)
@click.option(
    "--save_gpu_ram",
    is_flag=True,
    default=False,
    show_default=True,
    type=bool,
    help="This flag will put all the models to RAM instead of GPU RAM, and only put models to GPU when they are being used.",
)
@click.option(
    "--cpu",
    is_flag=True,
    default=False,
    show_default=True,
    type=bool,
    help="This flag will force the server to use cpu only.",
)
@click.option(
    "--focus",
    default="all",
    show_default=True,
    type=click.Choice([e.value for e in Focus]),
    help="""
Indicates which endpoints should we focus on, helpful if we only care about certain subset of features.
\n- all            |  will load all endpoints.
\n- sd             |  will load SD endpoints.
\n- sd.base        |  will only load basic SD endpoints, which means anime / inpainting / outpainting endpoints will not be loaded.
\n- sd.anime       |  will only load anime SD endpoints, which means basic / inpainting / outpainting endpoints will not be loaded.
\n- sd.inpainting  |  will only load inpainting / outpainting SD endpoints, which means basic / anime endpoints will not be loaded.
\n- sync           |  will only load 'sync' endpoints, which are relatively fast (e.g. sod, lama, captioning, harmonization, ...).
\n- control        |  will only load 'ControlNet' endpoints.
\n- pipeline       |  will only load 'Pipeline' endpoints.
\n-
""",
)
@click.option(
    "-r",
    "--reload",
    is_flag=True,
    default=False,
    show_default=True,
    type=bool,
    help="This flag represents the `reload` argument of `uvicorn.run`.",
)
@click.option(
    "-d",
    "--cache_dir",
    default="",
    show_default=True,
    type=str,
    help="Directory of the cache files.",
)
@click.option(
    "--disable_lazy",
    is_flag=True,
    default=False,
    show_default=True,
    type=bool,
    help="This flag will disable the `auto lazy loading` mode.",
)
def serve(
    *,
    port: int,
    save_gpu_ram: bool,
    cpu: bool,
    focus: str,
    reload: bool,
    cache_dir: str,
    disable_lazy: bool,
) -> None:
    increment = dict(auto_lazy_load=not disable_lazy)
    if save_gpu_ram:
        increment["save_gpu_ram"] = True
    if cpu:
        increment["cpu"] = True
    if focus != "all":
        increment["focus"] = focus
    cflearn_increment = {}
    if cache_dir:
        cflearn_increment["cache_dir"] = cache_dir
    with opt_env_context(increment):
        with OPT.opt_context(cflearn_increment):
            uvicorn.run(
                "cfcreator.apis.interface:app",
                host="0.0.0.0",
                port=port,
                reload=reload,
                reload_dirs=os.path.dirname(__file__) if reload else None,
            )
