from typing import Tuple

import numpy as np
import torch
from torch import nn

from .base import SpatialDims
from .utils import dims_after_pooling


class AffinityVAE(nn.Module):
    """Affinity regularized variational autoencoder.

    Parameters
    ----------
    input_shape : tuple
        A tuple representing the input shape of the data, e.g. (1, 64, 64) for
        images or (1, 64, 64, 64) for a volume with 1 channel.
    latent_dims : int
        The size of the latent representation.
    pose_dims : int
        The size of the pose representation.
    filters : tuple
        A tuple representing the filters for each convolutional layer.
    """

    def __init__(
        self,
        input_shape: Tuple[int] = (1, 64, 64),
        latent_dims: int = 8,
        pose_dims: int = 1,
        filters: Tuple[int] = (8, 16, 32, 64),
    ):
        super(AffinityVAE, self).__init__()

        channels = input_shape[0]
        spatial_dims = input_shape[1:]
        ndim = len(spatial_dims)

        if len(filters) != 4:
            raise ValueError(
                f"`filters` must be a tuple of length 4, got: {len(filters)}."
            )

        if ndim not in SpatialDims._value2member_map_:
            raise ValueError(
                f"`input_shape` must be have 2 or 3 dimensions, got: {ndim}."
            )

        if ndim == SpatialDims.TWO:
            conv = nn.Conv2d
            conv_T = nn.ConvTranspose2d
        elif ndim == SpatialDims.THREE:
            conv = nn.Conv3d
            conv_T = nn.ConvTranspose3d

        unflat_shape = tuple(
            [
                64,
            ]
            + [
                dims_after_pooling(ax, idx + 1)
                for idx, ax in enumerate(spatial_dims)
            ]
        )
        flat_shape = np.prod(unflat_shape)

        self.encoder = nn.Sequential(
            conv(channels, filters[0], 3, stride=2, padding=1),
            nn.ReLU(True),
            conv(filters[0], filters[1], 3, stride=2, padding=1),
            nn.ReLU(True),
            conv(filters[1], filters[2], 3, stride=2, padding=1),
            nn.ReLU(True),
            conv(filters[2], filters[3], 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dims + pose_dims, flat_shape),
            nn.Unflatten(-1, unflat_shape),
            conv_T(filters[3], filters[2], 3, stride=2),
            nn.ReLU(True),
            conv_T(filters[2], filters[1], 3, stride=2, padding=1),
            nn.ReLU(True),
            conv_T(filters[1], filters[0], 3, stride=2, padding=1),
            nn.ReLU(True),
            conv_T(filters[0], channels, 2, stride=2, padding=1),
        )

        self.mu = nn.Linear(flat_shape, latent_dims)
        self.log_var = nn.Linear(flat_shape, latent_dims)
        self.pose = nn.Linear(flat_shape, pose_dims)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        mu, log_var, pose = self.encode(x)
        z = self.reparameterise(mu, log_var)
        z_pose = torch.cat([pose, z], dim=-1)
        x = self.decode(z_pose)
        return x, z, z_pose, mu, log_var

    def reparameterise(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        pose = self.pose(encoded)
        return mu, log_var, pose

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
