import enum

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
