import pickle
from typing import List

from src.domain.graph import Graph
from src.repository.interface.repository import Repository


class TrainingDataRepository(Repository):
    def __init__(self, path='src/data/') -> None:
        super().__init__()
        self.path = path

    def save(self, dataset: List[Graph]) -> None:
        with open(self.path + 'dataset.pickle', 'wb') as file:
            pickle.dump(dataset, file)

    def get_all(self) -> List[Graph]:
        with open(self.path + 'dataset.pickle', 'rb') as file:
            dataset = pickle.load(file)

        return dataset
