from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Any


class Repository(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def save(self, filename: str, dataset_name: str) -> None:
        pass

    @abstractmethod
    def get_all_features_and_labels_from_separate_files(self) -> List[Tuple[Any, Any]]:
        pass
