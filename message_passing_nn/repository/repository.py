from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Any


class Repository(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def save(self, filename: str, dataset_name: str) -> None:
        pass

    @abstractmethod
    def get_all_data(self) -> List[Tuple[Any, Any, Any]]:
        pass
