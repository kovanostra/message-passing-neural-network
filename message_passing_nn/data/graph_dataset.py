from typing import List, Tuple

import torch as to
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self,
                 dataset: List[Tuple[to.Tensor, List[List[int]], to.Tensor]],
                 normalize_features: bool = True,
                 normalize_labels: bool = True) -> None:
        self.features = self._extract_features(dataset)
        self.labels = self._extract_labels(dataset)
        self.normalize_features = normalize_features
        self.normalize_labels = normalize_labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple:
        return self.features[index], self.labels[index]

    @staticmethod
    def _extract_features(dataset: List[Tuple[to.Tensor, List[List[int]], to.Tensor]]) -> List[
        Tuple[to.Tensor, to.Tensor]]:
        features = []
        for index in range(len(dataset)):
            number_of_nodes = dataset[index][0].shape[0]
            all_neighbors = to.zeros(number_of_nodes, number_of_nodes) - to.ones(number_of_nodes, number_of_nodes)
            for node_id in range(number_of_nodes):
                node_neighbors = dataset[index][1][node_id]
                all_neighbors[node_id, :len(node_neighbors)] = to.tensor(node_neighbors)
            features.append((dataset[index][0], all_neighbors))
        return features

    @staticmethod
    def _extract_labels(dataset: List[Tuple[to.Tensor, List[List[int]], to.Tensor]]) -> List[to.Tensor]:
        return [dataset[index][2] for index in range(len(dataset))]
