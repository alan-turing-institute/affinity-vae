import enum
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


# Abstract Decoder
class AbstractDecoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, x_pose: torch.Tensor) -> torch.Tensor:
        pass
