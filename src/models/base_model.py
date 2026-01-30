from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class BaseModel(nn.Module, ABC):
    train_keys: tuple[str, ...] = ("loss",)
    val_keys: tuple[str, ...] = ()

    def __init__(self):
        super().__init__()
        return

    # @abstractmethod
    def get_output(self, *args, **kwargs) -> Tensor:
        pass

    # @abstractmethod
    # @torch.inference_mode()
    # def predict(self, *args, **kwargs) -> Tensor:
    #     pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> dict[str, Tensor]:
        pass

    @abstractmethod
    @torch.inference_mode()
    def validate_batch(self, *args, **kwargs) -> dict[str, float]:
        pass

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device