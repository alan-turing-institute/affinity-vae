import torch


def set_device(gpu):
    device = torch.device(
        "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
    )
    if gpu and device == "cpu":
        print(
            "\nWARNING: no GPU available, running on CPU instead.\n",
            flush=True,
        )
    return device


def dims_after_pooling(start: int, n_pools: int) -> int:
    """Calculate the size of a layer after n pooling ops."""
    return start // (2**n_pools)
