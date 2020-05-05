from abc import ABCMeta, abstractmethod
from typing import Any, Tuple, List

import torch as to
from torch.utils.data import DataLoader


class Preprocessor(metaclass=ABCMeta):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def train_validation_test_split(dataset: Any,
                                    batch_size: int,
                                    validation_split: float,
                                    test_split: float) -> Tuple[DataLoader, DataLoader, DataLoader]:
        pass

    @staticmethod
    @abstractmethod
    def extract_data_dimensions(dataset: Any) -> Tuple:
        pass

    @staticmethod
    @abstractmethod
    def flatten(tensors: to.Tensor, desired_size: Any = 0) -> to.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def normalize(tensors: to.Tensor, device: str) -> to.Tensor:
        pass
