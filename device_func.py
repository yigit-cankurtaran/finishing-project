import torch


def device_func():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("using metal acceleration")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device
