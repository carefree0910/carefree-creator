OPT = dict(
    verbose=True,
    save_gpu_ram=False,
    use_cos=True,
)


def verbose() -> bool:
    return OPT["verbose"]


def save_gpu_ram() -> bool:
    return OPT["save_gpu_ram"]


def use_cos() -> bool:
    return OPT["use_cos"]


__all__ = [
    "OPT",
    "use_cos",
    "verbose",
    "save_gpu_ram",
]
