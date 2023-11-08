import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SpatialDims


def set_layer_dim(ndim):
    global CONV
    global TCONV
    global BNORM
    if ndim == SpatialDims.TWO:
        CONV = nn.Conv2d
        TCONV = nn.ConvTranspose2d
        BNORM = nn.BatchNorm2d
    elif ndim == SpatialDims.THREE:
        CONV = nn.Conv3d
        TCONV = nn.ConvTranspose3d
        BNORM = nn.BatchNorm3d
    else:
        logging.error("Data must be 2D or 3D.")
        exit(1)


class Encoder(nn.Module):
    def __init__(
        self,
        bottom_dim,
        latent_dims,
        pose_dims=0,
        capacity=None,
        depth=None,
        filters=None,
        bnorm=True,
    ):
        super(Encoder, self).__init__()

        # TODO these checks should be perfomed on the model level when abstract
        # TODO that way we don't have to require as many parameters of Encoder
        if capacity is None and filters is None:
            raise RuntimeError(
                "Pass either capacity or filters when definining avae.Encoder."
            )
        elif filters is not None and len(filters) != 0:
            if 0 in self.filters:
                raise RuntimeError("Filter list cannot contain zeros.")
            self.filters = filters
            if depth is not None:
                logging.WARNING(
                    "You've passed 'filters' parameter as well as 'depth'. Filters take"
                    " priority so 'depth' and 'capacity' will be disregarded."
                )
        elif capacity is not None:
            if depth is None:
                raise RuntimeError(
                    "When passing initial 'capacity' parameter in avae.Encoder,"
                    " provide 'depth' parameter too."
                )
            self.filters = [capacity * 2**x for x in range(depth)]
        else:
            raise RuntimeError(
                "You must provide either capacity or filters when definity ave.Encoder."
            )

        self.bnorm = bnorm
        self.pose = not (pose_dims == 0)

        # iteratively define convolution and batch normalisation layers
        self.conv_enc = nn.ModuleList()
        if self.bnorm:
            self.norm_enc = nn.ModuleList()

        for d in range(len(self.filters)):
            self.conv_enc.append(
                CONV(
                    in_channels=(self.filters[d - 1] if d != 0 else 1),
                    out_channels=self.filters[d],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            if self.bnorm:
                self.norm_enc.append(BNORM(self.filters[d]))

        # define fully connected layers
        ch = 1 if depth == 0 else self.filters[-1]  # allow for no conv layers
        self.fc_mu = nn.Linear(
            in_features=ch * np.prod(bottom_dim),
            out_features=latent_dims,
        )
        self.fc_logvar = nn.Linear(
            in_features=ch * np.prod(bottom_dim),
            out_features=latent_dims,
        )
        if self.pose:
            self.fc_pose = nn.Linear(
                in_features=ch * np.prod(bottom_dim),
                out_features=pose_dims,
            )

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
        x_mu : torch.Tensor (N, latent_dims)
            Mini-batch of outputs representing latent means, where N stands
            for the number of samples in the mini-batch
            and 'latent_dims' defines the number of latent dimensions.
        x_logvar : torch.Tensor (N, latent_dims)
            Mini-batch of outputs representing latent log of the variance,
            where N stands for the number of samples in
            the mini-batch and 'latent_dims' defines the number of latent
            dimensions.
        x_pose : torch.Tensor (N, pose_dims)
            Optional return if pose is True. Mini-batch of outputs
            representing pose capturing the within-class
            variance, where N stands for the number of samples in the
            mini-batch and 'pose_dims' defines the number of
            pose dimensions.
        """
        for d in range(len(self.filters)):
            if self.bnorm:
                x = self.norm_enc[d](F.relu(self.conv_enc[d](x)))
            else:
                x = F.relu(self.conv_enc[d](x))
        x = x.view(x.size(0), -1)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        if self.pose:
            x_pose = self.fc_pose(x)
            return x_mu, x_logvar, x_pose
        else:
            return x_mu, x_logvar


class Decoder(nn.Module):
    """Affinity decoder. Includes optional pose component merge.

    Parameters
    ----------
    capacity : int
        The capacity of the network - initial number of nodes doubled at each
        depth.
    depth : int
        The depth of the network - number of downsampling layers.
    bottom_dim: tuple (X, Y) or tuple (X, Y, Z)
        Tuple representing the size after downsampling for each image
        dimension X, Y and Z.
    latent_dims: int
        Number of bottleneck latent dimensions.
    pose: bool
        Determines whether pose component is on or off.
    pose_dims : int
        Number of bottleneck pose dimensions.
    """

    def __init__(
        self,
        bottom_dim,
        latent_dims,
        pose_dims=0,
        capacity=None,
        depth=None,
        filters=None,
        bnorm=True,
    ):
        super(Decoder, self).__init__()

        # TODO these checks should be perfomed on the model level when abstract
        # TODO that way we don't have to require as many parameters of Encoder
        if capacity is None and filters is None:
            raise RuntimeError(
                "Pass either capacity or filters when definining avae.Decoder."
            )
        elif filters is not None and len(filters) != 0:
            if 0 in self.filters:
                raise RuntimeError("Filter list cannot contain zeros.")
            self.filters = filters
            if depth is not None:
                logging.WARNING(
                    "You've passed 'filters' parameter as well as 'depth'. Filters take"
                    " priority so 'depth' and 'capacity' will be disregarded."
                )
        elif capacity is not None:
            if depth is None:
                raise RuntimeError(
                    "When passing initial 'capacity' parameter in avae.Encoder,"
                    " provide 'depth' parameter too."
                )
            self.filters = [capacity * 2**x for x in range(depth)]
        else:
            raise RuntimeError(
                "You must provide either capacity or filters when definity ave.Decoder."
            )

        self.bottom_dim = bottom_dim
        self.pose = not (pose_dims == 0)
        self.bnorm = bnorm

        #  iteratively define deconvolution and batch normalisation layers
        self.conv_dec = nn.ModuleList()
        if self.bnorm:
            self.norm_dec = nn.ModuleList()

        for d in reversed(range(len(self.filters))):
            self.conv_dec.append(
                TCONV(
                    in_channels=self.filters[d],
                    out_channels=(self.filters[d - 1] if d != 0 else 1),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            if self.bnorm and d != 0:
                self.norm_dec.append(BNORM(self.filters[d - 1]))

        # define fully connected layers
        self.ch = (
            1 if depth == 0 else self.filters[-1]
        )  # allow for no convolutions
        if self.pose:
            self.fc = nn.Linear(
                in_features=pose_dims + latent_dims,
                out_features=self.ch * np.prod(self.bottom_dim),
            )
        else:
            self.fc = nn.Linear(
                in_features=latent_dims,
                out_features=self.ch * np.prod(self.bottom_dim),
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
            x = self.fc(torch.cat((x, x_pose), -1))
        else:
            x = self.fc(x)
        x = x.view(x.size(0), self.ch, *self.bottom_dim)
        for d in range(len(self.filters) - 1):
            if self.bnorm:
                x = self.norm_dec[d](F.relu(self.conv_dec[d](x)))
            else:
                x = F.relu(self.conv_dec[d](x))
        x = torch.sigmoid(self.conv_dec[-1](x))
        return x


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
        bnorm=True,
    ):
        super(AffinityVAE, self).__init__()
        assert all(
            [int(x) == x for x in np.array(input_size) / (2**depth)]
        ), (
            "Input size not compatible with --depth. Input must be divisible "
            "by {}.".format(2**depth)
        )
        set_layer_dim(len(input_size))
        self.bottom_dim = tuple([int(i / (2**depth)) for i in input_size])
        self.pose = not (pose_dims == 0)

        self.encoder = Encoder(
            self.bottom_dim,
            latent_dims,
            pose_dims=pose_dims,
            capacity=capacity,
            depth=depth,
            bnorm=bnorm,
        )

        self.decoder = Decoder(
            self.bottom_dim,
            latent_dims,
            pose_dims=pose_dims,
            capacity=capacity,
            depth=depth,
            bnorm=bnorm,
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
        x_recon : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of outputs, where N stands for the number of samples in
            the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.
        latent_mu : torch.Tensor (N, latent_dims)
            Mini-batch of encoder outputs representing latent means, where N
            stands for the number of samples in the
            mini-batch and 'latent_dims' defines the number of latent
            dimensions.
        latent_logvar : torch.Tensor (N, latent_dims)
            Mini-batch of encoder outputs representing latent log of the
            variance, where N stands for the number of
            samples in the mini-batch and 'latent_dims' defines the number of
            latent dimensions.
        latent : torch.Tensor (N, latent_dims)
            Mini-batch of reparametrised encoder outputs, where N stands for
            the number of samples in the mini-batch
            and 'latent_dims' defines the number of latent dimensions.
        latent_pose : torch.Tensor (N, pose_dims)
            Optional return if pose is True. Mini-batch of encoder outputs
            representing pose capturing the within-class
            variance, where N stands for the number of samples in the
            mini-batch and 'pose_dims' defines the number of
            pose dimensions.

        """
        # encode
        if self.pose:
            latent_mu, latent_logvar, latent_pose = self.encoder(x)
        else:
            latent_mu, latent_logvar = self.encoder(x)
            latent_pose = None
        # reparametrise
        latent = self.sample(latent_mu, latent_logvar)
        # decode
        x_recon = self.decoder(latent, latent_pose)  # pose set to None if pd=0
        return x_recon, latent_mu, latent_logvar, latent, latent_pose

    def sample(self, mu, logvar):
        """Reparametrisation trick.

        Parameters
        ----------
        mu : torch.Tensor (N, latent_dims)
            Mini-batch of outputs representing latent means, where N stands
            for the number of samples in the mini-batch
            and 'latent_dims' defines the number of latent dimensions.
        logvar : torch.Tensor (N, latent_dims)
            Mini-batch of outputs representing latent log of the variance,
            where N stands for the number of samples in
            the mini-batch and 'latent_dims' defines the number of latent
            dimensions.

        Returns
        -------
        latent : torch.Tensor (N, latent_dims)
            Mini-batch of reparametrised encoder outputs, where N stands for
            the number of samples in the mini-batch
            and 'latent_dims' defines the number of latent dimensions.
        """
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
