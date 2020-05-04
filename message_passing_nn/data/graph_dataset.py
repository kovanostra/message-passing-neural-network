from typing import List, Tuple

import torch as to
import torchvision as tv
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self,
                 dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]],
                 normalize_features: bool = True,
                 normalize_labels: bool = True) -> None:
        self.features = self._to_list(dataset)
        self.labels = self._extract_labels(dataset)
        self.normalize_features = normalize_features
        self.normalize_labels = normalize_labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple:
        node_features, adjacency_matrix = self.features[index]
        labels = self.labels[index]
        if self.normalize_features:
            node_features = self._normalize(node_features)
        if self.normalize_labels:
            labels = self._normalize(labels)
        return (node_features, adjacency_matrix), labels

    @staticmethod
    def _to_list(dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]]) -> List[Tuple[to.Tensor, to.Tensor]]:
        return [(dataset[index][0], dataset[index][1]) for index in range(len(dataset))]

    @staticmethod
    def _extract_labels(dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]]) -> List[to.Tensor]:
        return [dataset[index][2] for index in range(len(dataset))]

    @staticmethod
    def _normalize(tensor: to.Tensor) -> to.Tensor:
        return to.div(tensor, to.max(tensor))
