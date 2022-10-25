OPT = dict(
    save_gpu_ram=False,
)


def save_gpu_ram() -> bool:
    return OPT["save_gpu_ram"]


__all__ = [
    "OPT",
    "save_gpu_ram",
]
