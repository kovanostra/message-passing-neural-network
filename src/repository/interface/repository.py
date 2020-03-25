from abc import ABCMeta, abstractmethod
from typing import List

from src.domain.graph import Graph


class Repository(metaclass=ABCMeta):
    def __init__(self):
        self.path = None

    @abstractmethod
    def save(self, dataset: List[Graph]) -> None:
        pass

    @abstractmethod
    def get_all(self) -> List[Graph]:
        pass
