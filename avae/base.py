import enum
from abc import ABC, abstractmethod

import torch


class SpatialDims(enum.IntEnum):
    TWO = 2
    THREE = 3


# Abstract Encoder
class AbstractEncoder(ABC):
    @abstractmethod
    def forward(self, x):
        pass


# Abstract Decoder
class AbstractDecoder(ABC):
    @abstractmethod
    def forward(self, x, x_pose):
        pass


# Abstract AffinityVAE
class AbstractAffinityVAE(ABC, torch.nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.init_model(**kwargs)

    @abstractmethod
    def init_model(self, **kwargs):
        pass

    def forward(self, x):
        mu, log_var, pose = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x = self.decoder(z, pose)
        return x, mu, log_var, z, pose

    def reparameterize(self, mu, log_var):
        # Your reparameterization code
        pass

    @staticmethod
    def set_device(gpu):
        # Your set_device code
        pass
