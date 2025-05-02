import logging
import torch
from abc import ABC, abstractmethod


class LocalUpdate(ABC):

    def __init__(self):
        self.logger = logging.getLogger(
            f"{__name__}-{self.__class__.__name__}"
        )

    @abstractmethod
    def update(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
    ):
        pass
