import enum
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SpatialDims(enum.IntEnum):
    TWO = 2
    THREE = 3


# Abstract AffinityVAE
class AbstractAffinityVAE(torch.nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu, log_var, pose = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x = self.decoder(z, pose)
        return x, mu, log_var, z, pose

    def reparameterize(self, mu, log_var):
        pass
