from abc import ABCMeta, abstractmethod
from typing import Any, Tuple, List

from torch.utils.data import DataLoader


class Preprocessor(metaclass=ABCMeta):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def train_validation_test_split(dataset: Any,
                                    batch_size: int,
                                    maximum_number_of_nodes: int,
                                    maximum_number_of_features: int,
                                    validation_split: float = 0.2,
                                    test_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        pass

    @staticmethod
    @abstractmethod
    def extract_data_dimensions(dataset: Any) -> Tuple:
        pass

    @staticmethod
    @abstractmethod
    def flatten(tensors: List[Any], desired_size: Any = 0) -> Any:
        pass
