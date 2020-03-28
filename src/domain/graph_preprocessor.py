from math import ceil
from typing import Any

import torch as to

from src.domain.graph import Graph
from src.domain.interface.preprocessor import Preprocessor


class GraphPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()

    def preprocess(self, dataset: Any, batches: int) -> Any:
        batch_length = self._calculate_batch_length(len(dataset), batches)
        dataset_in_batches = []
        for batch_index in range(0, len(dataset), batch_length):
            batch_is_complete = False
            batch = []
            data_index = batch_index
            while not batch_is_complete:
                batch.append(Graph(dataset[data_index][1], dataset[data_index][0], to.tensor([])))
                data_index += 1
                if len(batch) == batch_length or data_index == len(dataset):
                    batch_is_complete = True
            dataset_in_batches.append(batch)
        return dataset_in_batches

    @staticmethod
    def _calculate_batch_length(dataset_length: int, batches: int) -> int:
        if batches <= 1:
            batch_length = dataset_length
        elif dataset_length / 2 <= batches < dataset_length:
            batch_length = 2
        elif batches >= dataset_length:
            batch_length = 1
        else:
            batch_length = ceil(dataset_length / batches)
        return batch_length
