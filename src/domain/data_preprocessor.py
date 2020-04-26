import logging
from typing import Any, Tuple, List

import torch as to
from math import ceil
from torch.utils.data import DataLoader

from src.domain.graph import Graph
from src.domain.graph_dataset import GraphDataset
from src.domain.interface.preprocessor import Preprocessor


class DataPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()

    @staticmethod
    def train_validation_test_split(raw_dataset: Any,
                                    batch_size: int,
                                    validation_split: float = 0.2,
                                    test_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        test_index, validation_index = DataPreprocessor._get_validation_and_test_indexes(raw_dataset, test_split,
                                                                                         validation_split)
        training_data = DataLoader(GraphDataset(raw_dataset[:validation_index]), batch_size)
        validation_data = DataLoader(GraphDataset(raw_dataset[validation_index:test_index]), batch_size)
        test_data = DataLoader(GraphDataset(raw_dataset[test_index:]), batch_size)
        return training_data, validation_data, test_data

    @staticmethod
    def extract_initialization_graph(raw_dataset: Any) -> Graph:
        return Graph(raw_dataset[0][1], raw_dataset[0][0])

    @staticmethod
    def flatten(tensors: List[Any], desired_size: Any = 0) -> Any:
        flattened_tensor = tensors.view(-1)
        if 0 < desired_size != len(flattened_tensor):
            size_difference = abs(len(flattened_tensor) - desired_size)
            flattened_tensor = to.cat((flattened_tensor, to.zeros(size_difference)))
        return flattened_tensor

    @staticmethod
    def _get_validation_and_test_indexes(raw_dataset, test_split, validation_split):
        validation_index = int((1 - validation_split - test_split) * len(raw_dataset))
        test_index = int((1 - test_split) * len(raw_dataset))
        return test_index, validation_index

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
