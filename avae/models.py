import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from avae.base import AbstractAffinityVAE, AbstractDecoder, AbstractEncoder

from .base import SpatialDims

# Define abstract classes for Encoder, Decoder, and AffinityVAE (as shown in the previous code)


def set_layer_dim(ndim):
    if ndim == SpatialDims.TWO:
        return nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d
    elif ndim == SpatialDims.THREE:
        return nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d
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


def set_device(gpu):
    """Set the torch device to use for training and inference.

    Parameters
    ----------
    gpu: bool
        If True, the model will be trained on GPU.

    Returns
    -------
    device: torch.device

    """
    device = torch.device(
        "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
    )
    if gpu and device == "cpu":
        logging.warning(
            "\n\nWARNING: no GPU available, running on CPU instead.\n"
        )
    return device


#
# Concrete implementation of the AffinityVAE
class AffinityVAE(AbstractAffinityVAE):
    def __init__(self, encoder, decoder):
        super(AffinityVAE, self).__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu, log_var, pose = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x = self.decoder(z, pose)
        return x, mu, log_var, z, pose

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu
