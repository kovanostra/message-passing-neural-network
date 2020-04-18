from typing import List, Tuple, Any

from torch.utils.data import Dataset

from src.domain.graph import Graph


class GraphDataset(Dataset):
    def __init__(self, raw_dataset: List[Tuple[Any, Any]]) -> None:
        self.features = self.to_graph(raw_dataset)
        self.labels = self.extract_labels(raw_dataset)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple:
        return self.features[index], self.labels[index]

    @staticmethod
    def to_graph(raw_dataset: List[Tuple[Any, Any]]) -> List[Graph]:
        return [Graph(raw_dataset[index][1], raw_dataset[index][0]) for index in range(len(raw_dataset))]

    @staticmethod
    def extract_labels(raw_dataset: List[Tuple[Any, Any]]) -> List[Any]:
        return [raw_dataset[index][1] for index in range(len(raw_dataset))]
