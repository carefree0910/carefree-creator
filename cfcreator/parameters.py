OPT = dict(
    verbose=True,
    save_gpu_ram=False,
)


def verbose() -> bool:
    return OPT["verbose"]


def save_gpu_ram() -> bool:
    return OPT["save_gpu_ram"]


__all__ = [
    "OPT",
    "save_gpu_ram",
]
