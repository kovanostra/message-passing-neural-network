from abc import ABCMeta, abstractmethod

import numpy as np

from src.domain.graph import Graph


class Messenger(metaclass=ABCMeta):
    def __init__(self):
        self.time_steps = None

    @abstractmethod
    def compose_messages_from_nodes_to_targets(self, graph: Graph, messages: np.array) -> np.array:
        pass
