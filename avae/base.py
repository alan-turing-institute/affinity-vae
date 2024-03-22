import enum
import logging

import torch

from avae.decoders.base import AbstractDecoder
from avae.encoders.base import AbstractEncoder


class SpatialDims(enum.IntEnum):
    TWO = 2
    THREE = 3


# Abstract AffinityVAE
class AbstractAffinityVAE(torch.nn.Module):
    def __init__(
        self, encoder: AbstractEncoder, decoder: AbstractDecoder, **kwargs
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        mu, log_var, pose = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x = self.decoder(z, pose)
        return x, mu, log_var, z, pose

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor):
        raise NotImplementedError(
            "Reparameterize method must be implemented in child class."
        )


def set_layer_dim(
    ndim: SpatialDims | int,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    if ndim == SpatialDims.TWO:
        return torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d
    elif ndim == SpatialDims.THREE:
        return torch.nn.Conv3d, torch.nn.ConvTranspose3d, torch.nn.BatchNorm3d
    else:
        logging.error("Data must be 2D or 3D.")
        exit(1)


def dims_after_pooling(start: int, n_pools: int) -> int:
    """Calculate the size of a layer after n pooling ops.

    Parameters
    ----------
    start: int
        The size of the layer before pooling.
    n_pools: int
        The number of pooling operations.

    Returns
    -------
    int
        The size of the layer after pooling.


    """
    return start // (2**n_pools)
