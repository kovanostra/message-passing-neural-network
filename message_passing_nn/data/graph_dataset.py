from typing import List, Tuple

import torch as to
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, raw_dataset: List[Tuple[to.Tensor, to.Tensor]]) -> None:
        self.features = self._to_list(raw_dataset)
        self.labels = self._preprocess_labels(raw_dataset)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple:
        return self.features[index], self.labels[index]

    @staticmethod
    def _to_list(raw_dataset: List[Tuple[to.Tensor, to.Tensor]]) -> List[Tuple[to.Tensor, to.Tensor]]:
        return [(raw_dataset[index][0], raw_dataset[index][1].view(-1)) for index in range(len(raw_dataset))]

    def _preprocess_labels(self, raw_dataset: List[Tuple[to.Tensor, to.Tensor]]) -> List[to.Tensor]:
        labels = self._extract_labels(raw_dataset)
        labels_preprocessed = []
        for index in range(len(labels)):
            upper_triangular_indices = to.triu(to.ones_like(labels[index]) - to.eye(labels[index].size()[0],
                                                                                    labels[index].size()[1])) == 1
            labels_preprocessed.append(labels[index][upper_triangular_indices].float())
        return labels_preprocessed

    @staticmethod
    def _extract_labels(raw_dataset: List[Tuple[to.Tensor, to.Tensor]]) -> List[to.Tensor]:
        return [raw_dataset[index][1] for index in range(len(raw_dataset))]
