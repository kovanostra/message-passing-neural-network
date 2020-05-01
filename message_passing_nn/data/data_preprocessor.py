import logging
from typing import Tuple, List

import torch as to
from torch.utils.data import DataLoader

from message_passing_nn.data.graph_dataset import GraphDataset
from message_passing_nn.data.preprocessor import Preprocessor
from message_passing_nn.model.graph import Graph


class DataPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()

    @staticmethod
    def train_validation_test_split(raw_dataset: List[Tuple[to.Tensor, to.Tensor]],
                                    batch_size: int,
                                    validation_split: float = 0.2,
                                    test_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        test_index, validation_index = DataPreprocessor._get_validation_and_test_indexes(raw_dataset, test_split,
                                                                                         validation_split)
        training_data = DataLoader(GraphDataset(raw_dataset[:validation_index]), batch_size)
        if validation_split:
            validation_data = DataLoader(GraphDataset(raw_dataset[validation_index:test_index]), batch_size)
        else:
            validation_data = DataLoader(GraphDataset([]))
        if test_split:
            test_data = DataLoader(GraphDataset(raw_dataset[test_index:]), batch_size)
        else:
            test_data = DataLoader(GraphDataset([]))
        return training_data, validation_data, test_data

    @staticmethod
    def extract_initialization_graph(raw_dataset: List[Tuple[to.Tensor, to.Tensor]]) -> Graph:
        return Graph(raw_dataset[0][1], raw_dataset[0][0])

    @staticmethod
    def flatten(tensors: to.Tensor, desired_size: int = 0) -> to.Tensor:
        flattened_tensor = tensors.view(-1)
        if 0 < desired_size != len(flattened_tensor):
            flattened_tensor = DataPreprocessor._pad_zeros(flattened_tensor, desired_size)
        return flattened_tensor

    @staticmethod
    def _pad_zeros(flattened_tensor: to.Tensor, desired_size: int) -> to.Tensor:
        size_difference = abs(len(flattened_tensor) - desired_size)
        flattened_tensor = to.cat((flattened_tensor, to.zeros(size_difference)))
        return flattened_tensor

    @staticmethod
    def _get_validation_and_test_indexes(raw_dataset: List[Tuple[to.Tensor, to.Tensor]],
                                         test_split: float,
                                         validation_split: float) -> Tuple[int, int]:
        validation_index = int((1 - validation_split - test_split) * len(raw_dataset))
        test_index = int((1 - test_split) * len(raw_dataset))
        return test_index, validation_index

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
