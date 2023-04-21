import torch


def set_device(gpu):
    device = torch.device(
        "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
    )
    if gpu and device == "cpu":
        print("\nWARNING: no GPU available, running on CPU instead.\n")
    return device
