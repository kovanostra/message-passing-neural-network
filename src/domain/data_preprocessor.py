import logging
from typing import Any, Tuple

import torch as to
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
    def flatten(tensors: Any, desired_size: Any = 0) -> Any:
        flattened_tensor = tensors.view(-1)
        if 0 < desired_size != len(flattened_tensor):
            flattened_tensor = DataPreprocessor._pad_zeros(flattened_tensor, desired_size)
        return flattened_tensor

    @staticmethod
    def _pad_zeros(flattened_tensor: Any, desired_size: int) -> Any:
        size_difference = abs(len(flattened_tensor) - desired_size)
        flattened_tensor = to.cat((flattened_tensor, to.zeros(size_difference)))
        return flattened_tensor

    @staticmethod
    def _get_validation_and_test_indexes(raw_dataset, test_split, validation_split):
        validation_index = int((1 - validation_split - test_split) * len(raw_dataset))
        test_index = int((1 - test_split) * len(raw_dataset))
        return test_index, validation_index

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
