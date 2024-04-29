import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from avae.base import AbstractAffinityVAE

from .base import SpatialDims


def set_layer_dim(
    ndim: SpatialDims | int,
) -> tuple[nn.Module, nn.Module, nn.Module]:
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


def set_device(gpu: bool) -> torch.device:
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

        if self.encoder.pose != self.decoder.pose:
            logging.error("Encoder and decoder pose must be the same.")
            raise RuntimeError("Encoder and decoder pose must be the same.")

        self.pose = self.encoder.pose

    def forward(self, x):
        # encode
        if self.pose:
            latent_mu, latent_logvar, latent_pose = self.encoder(x)
        else:
            latent_mu, latent_logvar = self.encoder(x)
            latent_pose = None
        # reparametrise
        latent = self.reparametrise(latent_mu, latent_logvar)
        # decode
        x_recon, x_before_conv = self.decoder(latent, latent_pose)  # pose set to None if pd=0

        return x_recon, x_before_conv, latent_mu, latent_logvar, latent, latent_pose

    def reparametrise(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu
