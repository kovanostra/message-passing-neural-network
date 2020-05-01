from abc import ABCMeta, abstractmethod
from typing import Any, Tuple, List

from torch.utils.data import DataLoader

from message_passing_nn.model.graph import Graph


class Preprocessor(metaclass=ABCMeta):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def train_validation_test_split(raw_dataset: Any,
                                    batch_size: int,
                                    validation_split: float = 0.2,
                                    test_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        pass

    @staticmethod
    @abstractmethod
    def extract_initialization_graph(raw_dataset: Any) -> Graph:
        pass

    @staticmethod
    @abstractmethod
    def flatten(tensors: List[Any], desired_size: Any = 0) -> Any:
        pass
