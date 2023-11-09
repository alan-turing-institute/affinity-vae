# Concrete implementation of Decoder
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from avae.base import AbstractDecoder
from avae.models import SpatialDims, dims_after_pooling, set_layer_dim


class DecoderA(AbstractDecoder):
    def __init__(self, input_size, capacity, depth, latent_dims, pose_dims):
        super(DecoderA, self).__init__()
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

        _, conv_T, _ = set_layer_dim(ndim)

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
        if self.pose:
            return self.decoder(torch.cat([x_pose, x], dim=-1))
        else:
            return self.decoder(x)


class DecoderB(AbstractDecoder):
    """Affinity decoder. Includes optional pose component merge.

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

    def __init__(self, input_size, capacity, depth, latent_dims, pose_dims):
        super(DecoderB, self).__init__()
        self.c = capacity
        self.depth = depth
        self.pose = not (pose_dims == 0)

        assert all(
            [int(x) == x for x in np.array(input_size) / (2**depth)]
        ), (
            "Input size not compatible with --depth. Input must be divisible "
            "by {}.".format(2**depth)
        )
        _, TCONV, BNORM = set_layer_dim(len(input_size))
        self.bottom_dim = tuple([int(i / (2**depth)) for i in input_size])

        #  iteratively define deconvolution and batch normalisation layers
        self.conv_dec = nn.ModuleList()
        self.norm_dec = nn.ModuleList()
        prev_sh = self.c * depth
        for d in range(depth, 0, -1):
            sh = self.c * (d - 1) if d != 1 else 1
            self.conv_dec.append(
                TCONV(
                    in_channels=prev_sh,
                    out_channels=sh,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            self.norm_dec.append(BNORM(sh))
            prev_sh = sh

        # define fully connected layers
        self.chf = (
            1 if depth == 0 else self.c * depth
        )  # allow for no convolutions
        if self.pose:
            self.fc = nn.Linear(
                in_features=pose_dims + latent_dims,
                out_features=self.chf * np.prod(self.bottom_dim),
            )
        else:
            self.fc = nn.Linear(
                in_features=latent_dims,
                out_features=self.chf * np.prod(self.bottom_dim),
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
        x = x.view(x.size(0), self.chf, *self.bottom_dim)
        for d in range(self.depth - 1):
            x = self.norm_dec[d](F.relu(self.conv_dec[d](x)))
        x = torch.sigmoid(self.conv_dec[-1](x))
        return x
