from typing import List, Tuple

import torch as to
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, raw_dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]]) -> None:
        self.features = self._to_list(raw_dataset)
        self.labels = self._extract_labels(raw_dataset)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple:
        return self.features[index], self.labels[index]

    @staticmethod
    def _to_list(raw_dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]]) -> List[Tuple[to.Tensor, to.Tensor]]:
        return [(raw_dataset[index][0], raw_dataset[index][1]) for index in range(len(raw_dataset))]

    @staticmethod
    def _extract_labels(raw_dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]]) -> List[to.Tensor]:
        return [raw_dataset[index][2] for index in range(len(raw_dataset))]
