import os
import click
import uvicorn

from cfcreator import opt_env_context


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
    "--focus",
    default="all",
    show_default=True,
    type=click.Choice(["all", "sd", "sd.base", "sd.anime", "sd.inpainting"]),
    help="""
Indicates which endpoints should we focus on, helpful if we only care about certain subset of features.
\n- all            |  will load all endpoints.
\n- sd             |  will load SD endpoints.
\n- sd.base        |  will only load basic SD endpoints, which means anime / inpainting / outpainting endpoints will not be loaded.
\n- sd.anime       |  will only load anime SD endpoints, which means basic / inpainting / outpainting endpoints will not be loaded.
\n- sd.inpainting  |  will only load inpainting / outpainting SD endpoints, which means basic / anime endpoints will not be loaded.
\n-
""",
)
def serve(*, port: int, save_gpu_ram: bool, focus: str):
    increment = {}
    if save_gpu_ram:
        increment["save_gpu_ram"] = True
    if focus != "all":
        increment["focus"] = focus
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    with opt_env_context(increment):
        uvicorn.run(
            "apis.interface:app",
            host="0.0.0.0",
            port=port,
            reload=True,
            reload_dirs=root,
        )
