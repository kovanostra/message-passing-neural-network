from abc import ABCMeta, abstractmethod
from typing import List

from src.domain.graph import Graph


class Repository(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def get_all(self) -> List[Graph]:
        pass
