import logging

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from avae.encoders.base import AbstractEncoder
from avae.models import dims_after_pooling, set_layer_dim


class Encoder(AbstractEncoder):
    """Affinity encoder. Includes optional pose component in the architecture.

    Parameters
    ----------
    input_size: tuple (X, Y) or tuple (X, Y, Z)
        Tuple representing the size of the data for each image
        dimension X, Y and Z.
    latent_dims: int
        Number of bottleneck latent dimensions.
    pose_dims : int
        Number of bottleneck pose dimensions.
    capacity : int (None)
        The capacity of the network - initial number of nodes doubled at each
        depth.
    depth : int (None)
        The depth of the network - number of downsampling layers.
    filters : list : [int] (None)
        List of filter sizes, where len(filters) becomes network depth.
    bnorm : bool (True)
        If True, turns BatchNormalisation on.
    """

    def __init__(
        self,
        input_size,
        capacity=None,
        depth=None,
        latent_dims=8,
        pose_dims=0,
        filters=None,
        bnorm=True,
    ):

        super(Encoder, self).__init__()
        self.filters = []
        if capacity is None and filters is None:
            raise RuntimeError(
                "Pass either capacity or filters when definining avae.Encoder."
            )
        elif filters is not None and len(filters) != 0:
            if 0 in filters:
                raise RuntimeError("Filter list cannot contain zeros.")
            self.filters = filters
            if depth is not None:
                logging.warning(
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

        assert all(
            [int(x) == x for x in np.array(input_size) / (2**depth)]
        ), (
            "Input size not compatible with --depth. Input must be divisible "
            "by {}.".format(2**depth)
        )

        bottom_dim = tuple(
            [int(i / (2 ** len(self.filters))) for i in input_size]
        )
        self.bnorm = bnorm
        self.pose = not (pose_dims == 0)

        # define layer dimensions
        CONV, TCONV, BNORM = set_layer_dim(len(input_size))

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


class EncoderA(AbstractEncoder):
    def __init__(self, input_size, capacity, depth, latent_dims, pose_dims):
        super(EncoderA, self).__init__()
        self.pose = not (pose_dims == 0)

        assert all(
            [int(x) == x for x in np.array(input_size) / (2**depth)]
        ), (
            "Input size not compatible with --depth. Input must be divisible "
            "by {}.".format(2**depth)
        )
        filters = [capacity * 2**x for x in range(depth)]

        unflat_shape = tuple(
            [
                filters[-1],
            ]
            + [dims_after_pooling(ax, depth) for ax in input_size]
        )
        flat_shape = np.prod(unflat_shape)

        ndim = len(unflat_shape[1:])

        conv, _, _ = set_layer_dim(ndim)

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
        self.pose_fc = nn.Linear(flat_shape, pose_dims)

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        pose = self.pose_fc(encoded)
        return mu, log_var, pose


class EncoderB(AbstractEncoder):
    """Affinity encoder. Includes optional pose component in the architecture.

    Parameters
    ----------
    capacity : int
        The capacity of the network - initial number of nodes doubled at each
        depth.
    depth : int
        The depth of the network - number of downsampling layers.
    latent_dims: int
        Number of bottleneck latent dimensions.
    pose_dims : int
        Number of bottleneck pose dimensions.
    """

    def __init__(self, input_size, capacity, depth, latent_dims, pose_dims=0):
        super(EncoderB, self).__init__()

        assert all(
            [int(x) == x for x in np.array(input_size) / (2**depth)]
        ), (
            "Input size not compatible with --depth. Input must be divisible "
            "by {}.".format(2**depth)
        )
        CONV, _, BNORM = set_layer_dim(len(input_size))
        self.bottom_dim = tuple([int(i / (2**depth)) for i in input_size])

        c = capacity
        self.depth = depth
        self.pose = not (pose_dims == 0)

        # iteratively define convolution and batch normalisation layers
        self.conv_enc = nn.ModuleList()
        self.norm_enc = nn.ModuleList()
        prev_sh = 1
        for d in range(depth):
            sh = c * (d + 1)
            self.conv_enc.append(
                CONV(
                    in_channels=prev_sh,
                    out_channels=sh,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
            )
            self.norm_enc.append(BNORM(sh))
            prev_sh = sh

        # define fully connected layers
        chf = 1 if depth == 0 else c * depth  # allow for no conv layers
        self.fc_mu = nn.Linear(
            in_features=chf * np.prod(self.bottom_dim),
            out_features=latent_dims,
        )
        self.fc_logvar = nn.Linear(
            in_features=chf * np.prod(self.bottom_dim),
            out_features=latent_dims,
        )
        if self.pose:
            self.fc_pose = nn.Linear(
                in_features=chf * np.prod(self.bottom_dim),
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
        for d in range(self.depth):
            x = self.norm_enc[d](F.relu(self.conv_enc[d](x)))
        x = x.view(x.size(0), -1)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        if self.pose:
            x_pose = self.fc_pose(x)
            return x_mu, x_logvar, x_pose
        else:
            return x_mu, x_logvar