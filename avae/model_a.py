import numpy as np
import torch
import torch.nn as nn

from .base import SpatialDims


class Encoder(nn.Module):
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
        encoded = self.encoder(x)
        mu = self.mu(encoded)

        log_var = self.log_var(encoded)
        pose = self.pose(encoded)
        return mu, log_var, pose


class Decoder(nn.Module):
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

        ## create a variable that tells us if pose is activated
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
        if self.pose :
            return self.decoder(torch.cat([x_pose, x], dim=-1))
        else:
            return self.decoder(x)


class AffinityVAE(nn.Module):
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
        capacity = 8
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
    device = torch.device(
        "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
    )
    if gpu and device == "cpu":
        print("\nWARNING: no GPU available, running on CPU instead.\n")
    return device


def dims_after_pooling(start: int, n_pools: int) -> int:
    """Calculate the size of a layer after n pooling ops."""
    return start // (2**n_pools)
