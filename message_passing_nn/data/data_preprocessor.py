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
    def train_validation_test_split(raw_dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]],
                                    batch_size: int,
                                    maximum_number_of_nodes: int,
                                    maximum_number_of_features: int,
                                    validation_split: float = 0.2,
                                    test_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        preprocessed_dataset = DataPreprocessor._preprocess_dataset_dimensions(raw_dataset,
                                                                               maximum_number_of_nodes,
                                                                               maximum_number_of_features)
        test_index, validation_index = DataPreprocessor._get_validation_and_test_indexes(raw_dataset,
                                                                                         test_split,
                                                                                         validation_split)
        training_data = DataLoader(GraphDataset(preprocessed_dataset[:validation_index]), batch_size)
        if validation_split:
            validation_data = DataLoader(GraphDataset(preprocessed_dataset[validation_index:test_index]), batch_size)
        else:
            validation_data = DataLoader(GraphDataset([]))
        if test_split:
            test_data = DataLoader(GraphDataset(preprocessed_dataset[test_index:]), batch_size)
        else:
            test_data = DataLoader(GraphDataset([]))
        return training_data, validation_data, test_data

    @staticmethod
    def extract_initialization_graph(raw_dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]]) -> Graph:
        adjacency_matrix = raw_dataset[0][1]
        node_features = raw_dataset[0][0]
        return Graph(adjacency_matrix, node_features)

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
    def _get_validation_and_test_indexes(raw_dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]],
                                         test_split: float,
                                         validation_split: float) -> Tuple[int, int]:
        validation_index = int((1 - validation_split - test_split) * len(raw_dataset))
        test_index = int((1 - test_split) * len(raw_dataset))
        return test_index, validation_index

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')

    @staticmethod
    def _preprocess_dataset_dimensions(raw_dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]],
                                       maximum_number_of_nodes: int,
                                       maximum_number_of_features: int) \
            -> List[Tuple[to.Tensor, to.Tensor, to.Tensor]]:

        adjacency_matrix_max_size, features_max_size, labels_max_size = DataPreprocessor._get_maximum_dataset_sizes(
            maximum_number_of_nodes, raw_dataset)

        preprocessed_dataset = DataPreprocessor._equalize_sizes(adjacency_matrix_max_size,
                                                                features_max_size,
                                                                labels_max_size,
                                                                maximum_number_of_features,
                                                                maximum_number_of_nodes,
                                                                raw_dataset)
        return preprocessed_dataset

    @staticmethod
    def _equalize_sizes(adjacency_matrix_max_size: List[int],
                        features_max_size: List[int],
                        labels_max_size: List[int],
                        maximum_number_of_features: int,
                        maximum_number_of_nodes: int,
                        raw_dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]]) \
            -> List[Tuple[to.Tensor, to.Tensor, to.Tensor]]:
        preprocessed_dataset = []
        for features, adjacency_matrix, labels in raw_dataset:
            if -1 < maximum_number_of_nodes < adjacency_matrix.size()[0]:
                continue
            if maximum_number_of_features < 0:
                maximum_number_of_features = features.size()[1]
            features_preprocessed = to.zeros((features_max_size[0], maximum_number_of_features))
            adjacency_matrix_preprocessed = to.zeros((adjacency_matrix_max_size[0], adjacency_matrix_max_size[1]))
            labels_preprocessed = to.zeros((labels_max_size[0]))

            features_preprocessed[:features.size()[0], :maximum_number_of_features] = features[:,
                                                                                      :maximum_number_of_features]
            adjacency_matrix_preprocessed[:adjacency_matrix.size()[0], :adjacency_matrix.size()[1]] = adjacency_matrix
            labels_preprocessed[:labels.size()[0]] = labels
            preprocessed_dataset.append((features_preprocessed, adjacency_matrix_preprocessed, labels_preprocessed))
        return preprocessed_dataset

    @staticmethod
    def _get_maximum_dataset_sizes(maximum_number_of_nodes: int,
                                   raw_dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]]) \
            -> Tuple[List, List, List]:
        features_max_size = [0, 0]
        adjacency_matrix_max_size = [0, 0]
        labels_max_size = [0]
        for features, adjacency_matrix, labels in raw_dataset:
            if 0 < maximum_number_of_nodes < adjacency_matrix.size()[0]:
                continue
            if features.size()[0] > features_max_size[0]:
                features_max_size[0] = features.size()[0]
            if features.size()[1] > features_max_size[1]:
                features_max_size[1] = features.size()[1]
            if adjacency_matrix.size()[0] > adjacency_matrix_max_size[0]:
                adjacency_matrix_max_size = adjacency_matrix.size()
            if labels.size()[0] > labels_max_size[0]:
                labels_max_size = labels.size()
        return adjacency_matrix_max_size, features_max_size, labels_max_size
