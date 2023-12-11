import enum
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


# Abstract Decoder
class AbstractDecoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, x, x_pose):
        pass
