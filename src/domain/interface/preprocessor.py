from abc import ABCMeta, abstractmethod
from typing import Any


class Preprocessor(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def preprocess(self, dataset: Any, batches: int) -> Any:
        pass
