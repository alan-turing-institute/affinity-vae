import logging

import numpy as np
import torch
import torch.nn as nn

from .base import SpatialDims


class Encoder(nn.Module):
    """Affinity encoder. Includes optional pose component in the architecture.

    Parameters
    ----------
    depth : int
        The depth of the network - number of downsampling layers.
    filters: list
        List of filters sizes for each layer.
    latent_dims: int
        Number of  latent dimensions.
    pose_dims: int
        Number of pose dimensions.
    unflat_shape : tuple
           Shape of the input tensor.
    flat_shape : int
           Size of the flattened tensor.
    conv : nn.Module
        Convolutional layer to use.
    """

    def __init__(
        self,
        depth,
        filters,
        latent_dims,
        pose_dims,
        unflat_shape,
        flat_shape,
        conv,
    ):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential()

        input_channel = 1
        for d in range(len(filters)):
            self.encoder.append(
                conv(
                    in_channels=input_channel,
                    out_channels=filters[d],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            self.encoder.append(nn.ReLU(True))
            input_channel = filters[d]

        self.encoder.append(nn.Flatten())
        self.mu = nn.Linear(flat_shape, latent_dims)
        self.log_var = nn.Linear(flat_shape, latent_dims)
        self.pose = nn.Linear(flat_shape, pose_dims)

    def forward(self, x):
        """Encoder forward pass.

        Parameters
        ----------
        x : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of inputs, where N stands for the number of samples in
            the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.

        Returns
        -------
        mu : torch.Tensor (N, latent_dims)
            Mini-batch of outputs representing latent means, where N stands
            for the number of samples in the mini-batch
            and 'latent_dims' defines the number of latent dimensions.
        logvar : torch.Tensor (N, latent_dims)
            Mini-batch of outputs representing latent log of the variance,
            where N stands for the number of samples in
            the mini-batch and 'latent_dims' defines the number of latent
            dimensions.
        pose : torch.Tensor (N, pose_dims)
            Optional return if pose is True. Mini-batch of outputs
            representing pose capturing the within-class
            variance, where N stands for the number of samples in the
            mini-batch and 'pose_dims' defines the number of
            pose dimensions.
        """
        encoded = self.encoder(x)
        mu = self.mu(encoded)

        log_var = self.log_var(encoded)
        pose = self.pose(encoded)
        return mu, log_var, pose


class Decoder(nn.Module):
    """Affinity decoder. Includes optional pose component merge.

    Parameters
    ----------
    depth : int
        The depth of the network - number of downsampling layers.
    filters: list
        List of filters sizes for each layer.
    latent_dims: int
        Number of  latent dimensions.
    pose_dims: int
        Number of pose dimensions.
    unflat_shape : tuple
        Shape of the input tensor.
    flat_shape : int
        Size of the flattened tensor.
    conv_T : nn.Module
        Transposed Convolutional layer to use.
    """

    def __init__(
        self,
        depth,
        filters,
        latent_dims,
        pose_dims,
        unflat_shape,
        flat_shape,
        conv_T,
    ):
        super(Decoder, self).__init__()

        # create a variable that tells us if pose is activated
        self.pose = not (pose_dims == 0)

        self.decoder = nn.Sequential()

        self.decoder.append(nn.Linear(latent_dims + pose_dims, flat_shape))
        self.decoder.append(nn.Unflatten(-1, unflat_shape))

        for d in reversed(range(len(filters))):
            if d != 0:
                if not d == len(filters) - 1:
                    self.decoder.append(
                        conv_T(
                            in_channels=filters[d],
                            out_channels=filters[d - 1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        )
                    )
                else:
                    self.decoder.append(
                        conv_T(
                            in_channels=filters[d],
                            out_channels=filters[d - 1],
                            kernel_size=3,
                            stride=2,
                            padding=0,
                        )
                    )

                self.decoder.append(nn.ReLU(True))
        self.decoder.append(
            conv_T(
                in_channels=filters[0],
                out_channels=1,
                kernel_size=2,
                stride=2,
                padding=1,
            )
        )

    def forward(self, x, x_pose):
        """Decoder forward pass.

        Parameters
        ----------
        x : torch.Tensor (N, latent_dims)
            Mini-batch of reparametrised encoder outputs, where N stands for
            the number of samples in the mini-batch and
            'latent_dims' defines the number of latent dimensions.
        x_pose : torch.Tensor (N, pose_dims)
            Mini-batch of outputs representing pose capturing the within-class
            variance, where N stands for the number
            of samples in the mini-batch and 'pose_dims' defines the number of
            pose dimensions.

        Returns
        -------
        x : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of outputs, where N stands for the number of samples in
            the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.

        """

        if self.pose:
            return self.decoder(torch.cat([x_pose, x], dim=-1))
        else:
            return self.decoder(x)


class AffinityVAE(nn.Module):
    """Affinity regularised Variational Autoencoder with an optional
    within-class variance encoding pose component.

    Parameters
    ----------
    capacity : int
        The capacity of the network - initial number of nodes doubled at each
        depth.
    depth : int
        The depth of the network - number of downsampling layers.
    input_size: tuple (X, Y) or tuple (X, Y, Z)
        Tuple representing the size of the input for each dimension  X, Y and
        Z.
    latent_dims: int
        Number of bottleneck latent dimensions.
    pose: bool
        Determines whether pose component is on or off.
    pose_dims : int
        Number of bottleneck pose dimensions.
    """

    def __init__(
        self,
        capacity,
        depth,
        input_size,
        latent_dims,
        pose_dims=0,
    ):
        super(AffinityVAE, self).__init__()
        assert all(
            [int(x) == x for x in np.array(input_size) / (2**depth)]
        ), (
            "Input size not compatible with --depth. Input must be divisible "
            "by {}.".format(2**depth)
        )
        self.bottom_dim = tuple([int(i / (2**depth)) for i in input_size])
        self.pose = not (pose_dims == 0)
        self.filters = [capacity * 2**x for x in range(depth)]

        self.unflat_shape = tuple(
            [
                self.filters[-1],
            ]
            + [dims_after_pooling(ax, depth) for ax in input_size]
        )
        self.flat_shape = np.prod(self.unflat_shape)

        ndim = len(self.unflat_shape[1:])

        conv = nn.Conv3d
        conv_T = nn.ConvTranspose3d
        if ndim == SpatialDims.TWO:
            conv = nn.Conv2d
            conv_T = nn.ConvTranspose2d
        elif ndim == SpatialDims.THREE:
            conv = nn.Conv3d
            conv_T = nn.ConvTranspose3d

        self.encoder = Encoder(
            depth,
            self.filters,
            latent_dims,
            pose_dims,
            self.unflat_shape,
            self.flat_shape,
            conv,
        )

        self.decoder = Decoder(
            depth,
            self.filters,
            latent_dims,
            pose_dims,
            self.unflat_shape,
            self.flat_shape,
            conv_T,
        )

    def forward(self, x):
        """AffinityVAE forward pass.

        Parameters
        ----------
        x : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of inputs, where N stands for the number of samples in
            the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.

        Returns
        -------
        x : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of outputs, where N stands for the number of samples in
            the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.
        mu : torch.Tensor (N, latent_dims)
            Mini-batch of encoder outputs representing latent means, where N
            stands for the number of samples in the
            mini-batch and 'latent_dims' defines the number of latent
            dimensions.
        logvar : torch.Tensor (N, latent_dims)
            Mini-batch of encoder outputs representing latent log of the
            variance, where N stands for the number of
            samples in the mini-batch and 'latent_dims' defines the number of
            latent dimensions.
        z : torch.Tensor (N, latent_dims)
            Mini-batch of reparametrised encoder outputs, where N stands for
            the number of samples in the mini-batch
            and 'latent_dims' defines the number of latent dimensions.
        pose : torch.Tensor (N, pose_dims)
            Optional return if pose is True. Mini-batch of encoder outputs
            representing pose capturing the within-class
            variance, where N stands for the number of samples in the
            mini-batch and 'pose_dims' defines the number of
            pose dimensions.

        """
        mu, log_var, pose = self.encoder(x)
        z = self.reparameterise(mu, log_var)
        x = self.decoder(z, pose)
        return x, mu, log_var, z, pose

    def reparameterise(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        if self.training:

            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu


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
