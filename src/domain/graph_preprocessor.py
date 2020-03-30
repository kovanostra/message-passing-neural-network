import logging
from math import ceil
from typing import Any

from src.domain.graph import Graph
from src.domain.interface.preprocessor import Preprocessor


class GraphPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()

    def preprocess(self, dataset: Any, batches: int) -> Any:
        self.get_logger().info("Started preprocessing data")
        batch_length = self._calculate_batch_length(len(dataset), batches)
        dataset_in_batches = []
        for batch_index in range(0, len(dataset), batch_length):
            batch_is_complete = False
            batch = []
            data_index = batch_index
            while not batch_is_complete:
                batch.append(Graph(dataset[data_index][1], dataset[data_index][0]))
                data_index += 1
                if len(batch) == batch_length or data_index == len(dataset):
                    batch_is_complete = True
            dataset_in_batches.append(batch)
        self.get_logger().info("Finished preprocessing data into " + str(len(dataset_in_batches)) + " batches")
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

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
