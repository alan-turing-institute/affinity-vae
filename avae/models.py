import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from avae.base import AbstractAffinityVAE
from avae.decoders.decoders import Decoder, DecoderA, DecoderB
from avae.decoders.differentiable import GaussianSplatDecoder
from avae.encoders.encoders import Encoder, EncoderA, EncoderB

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


def build_model(
    model_type: str,
    input_shape: tuple,
    channels: int,
    depth: int,
    lat_dims: int,
    pose_dims: int,
    bnorm_encoder: bool,
    bnorm_decoder: bool,
    n_splats: int,
    gsd_conv_layers: int,
    device: torch.device,
    filters: list | None = None,
):
    """Create the AffinityVAE model.

    Parameters
    ----------
    model_type : str
        The type of model to create. Must be one of : a, b, u or gsd.
    input_shape : tuple
        The size of the input.
    channels : int
        The number of channels in the model.
    depth : int
        The depth of the model.
    lat_dims : int
        The number of latent dimensions.
    pose_dims : int
        The number of pose dimensions.
    bnorm_encoder : bool
        Whether to use batch normalisation in the encoder.
    bnorm_decoder : bool
        Whether to use batch normalisation in the decoder.
    n_splats : int
        The number of splats in the Gaussian Splat Decoder.
    gsd_conv_layers : int
        The number of convolutional layers in the Gaussian Splat Decoder.
    device : torch.device
        The device to use for training and inference.
    filters : list or None
        The filters to use in the model.

    """

    if filters is not None:
        filters = np.array(
            np.array(filters).replace(" ", "").split(","), dtype=np.int64
        )

    if model_type == "a":
        encoder = EncoderA(
            input_shape,
            channels,
            depth,
            lat_dims,
            pose_dims,
            bnorm=bnorm_encoder,
        )
        decoder = DecoderA(
            input_shape,
            channels,
            depth,
            lat_dims,
            pose_dims,
            bnorm=bnorm_decoder,
        )
    elif model_type == "b":
        encoder = EncoderB(input_shape, channels, depth, lat_dims, pose_dims)
        decoder = DecoderB(input_shape, channels, depth, lat_dims, pose_dims)
    elif model_type == "u":
        encoder = Encoder(
            input_shape=input_shape,
            capacity=channels,
            filters=filters,
            depth=depth,
            latent_dims=lat_dims,
            pose_dims=pose_dims,
            bnorm=bnorm_encoder,
        )
        decoder = Decoder(
            input_shape=input_shape,
            capacity=channels,
            filters=filters,
            depth=depth,
            latent_dims=lat_dims,
            pose_dims=pose_dims,
            bnorm=bnorm_decoder,
        )
    elif model_type == "gsd":
        encoder = EncoderA(
            input_shape,
            channels,
            depth,
            lat_dims,
            pose_dims,
            bnorm=bnorm_encoder,
        )
        decoder = GaussianSplatDecoder(
            input_shape,
            n_splats=n_splats,
            latent_dims=lat_dims,
            output_channels=gsd_conv_layers,
            device=device,
            pose_dims=pose_dims,
        )
    else:
        raise ValueError(
            "Invalid model type",
            model_type,
            "must be one of : a, b, u or gsd",
        )

    vae = AffinityVAE(encoder, decoder)

    return vae


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
        if self.decoder.__class__.__name__ == 'GaussianSplatDecoder':
            x_recon, x_before_conv = self.decoder(
                latent, latent_pose
            )  # pose set to None if pd=0
        else:
            x_recon = self.decoder(latent, latent_pose)

        return (
            x_recon,
            x_before_conv,
            latent_mu,
            latent_logvar,
            latent,
            latent_pose,
        )

    def reparametrise(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu
