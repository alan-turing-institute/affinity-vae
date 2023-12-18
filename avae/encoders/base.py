import enum
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


# Abstract Encoder
class AbstractEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass
