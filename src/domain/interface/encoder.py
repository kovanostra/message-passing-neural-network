from abc import ABCMeta, abstractmethod

import numpy as np

from src.domain.graph import Graph


class Encoder(metaclass=ABCMeta):
    @abstractmethod
    def encode(self, graph: Graph) -> np.array:
        pass
