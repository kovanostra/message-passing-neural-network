from typing import List, Tuple

import torch as to
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
        return self.features[index], self.labels[index]

    @staticmethod
    def _to_list(dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]]) -> List[Tuple[to.Tensor, to.Tensor]]:
        return [(dataset[index][0], dataset[index][1]) for index in range(len(dataset))]

    @staticmethod
    def _extract_labels(dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]]) -> List[to.Tensor]:
        return [dataset[index][2] for index in range(len(dataset))]

