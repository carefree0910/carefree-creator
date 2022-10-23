OPT = dict(
    save_memory=False,
)

def save_memory() -> bool:
    return OPT["save_memory"]


__all__ = [
    "OPT",
    "save_memory",
]
