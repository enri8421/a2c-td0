import typing

import torch

type DeviceType = typing.Literal["auto", "cpu", "cuda", "mps"]


def choose_device(device: DeviceType) -> typing.Literal["cpu", "cuda", "mps"]:
    """Resolve the execution device."""

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this machine; use --device cpu or auto")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available on this machine; use --device cpu or auto")
    return device
