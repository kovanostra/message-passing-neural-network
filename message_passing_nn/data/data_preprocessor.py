import logging
from typing import Tuple, List

import torch as to
from torch import nn
from torch.utils.data import DataLoader

from message_passing_nn.data.graph_dataset import GraphDataset
from message_passing_nn.data.preprocessor import Preprocessor


class DataPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()

    def train_validation_test_split(self,
                                    dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]],
                                    batch_size: int,
                                    maximum_number_of_nodes: int,
                                    maximum_number_of_features: int,
                                    validation_split: float = 0.2,
                                    test_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        test_index, validation_index = self._get_validation_and_test_indexes(dataset,
                                                                             validation_split,
                                                                             test_split)
        training_data = DataLoader(GraphDataset(dataset[:validation_index]), batch_size)
        if validation_split:
            validation_data = DataLoader(GraphDataset(dataset[validation_index:test_index]), batch_size)
        else:
            validation_data = DataLoader(GraphDataset([]))
        if test_split:
            test_data = DataLoader(GraphDataset(dataset[test_index:]), batch_size)
        else:
            test_data = DataLoader(GraphDataset([]))
        return training_data, validation_data, test_data

    def equalize_dataset_dimensions(self,
                                    dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]],
                                    maximum_number_of_nodes: int,
                                    maximum_number_of_features: int) \
            -> List[Tuple[to.Tensor, to.Tensor, to.Tensor]]:

        adjacency_matrix_max_size, features_max_size, labels_max_size = self._get_maximum_data_size(dataset,
                                                                                                    maximum_number_of_nodes)

        preprocessed_dataset = self._equalize_sizes(dataset,
                                                    adjacency_matrix_max_size,
                                                    features_max_size,
                                                    labels_max_size,
                                                    maximum_number_of_nodes,
                                                    maximum_number_of_features)
        return preprocessed_dataset

    @staticmethod
    def extract_data_dimensions(dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]]) -> Tuple:
        node_features_size = dataset[0][0].size()
        adjacency_matrix_size = dataset[0][1].size()
        labels_size = dataset[0][2].size()
        return node_features_size, adjacency_matrix_size, labels_size

    @staticmethod
    def flatten(tensors: to.Tensor, desired_size: int = 0) -> to.Tensor:
        flattened_tensor = tensors.view(-1)
        if 0 < desired_size != len(flattened_tensor):
            flattened_tensor = DataPreprocessor._pad_zeros(flattened_tensor, desired_size)
        return flattened_tensor

    @staticmethod
    def normalize(tensor: to.Tensor) -> to.Tensor:
        if tensor.size()[0] > 1:
            normalizer = nn.BatchNorm1d(tensor.size()[1], affine=False)
            return normalizer(tensor)
        else:
            return tensor

    @staticmethod
    def _pad_zeros(flattened_tensor: to.Tensor, desired_size: int) -> to.Tensor:
        size_difference = abs(len(flattened_tensor) - desired_size)
        flattened_tensor = to.cat((flattened_tensor, to.zeros(size_difference)))
        return flattened_tensor

    @staticmethod
    def _get_validation_and_test_indexes(dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]],
                                         validation_split: float,
                                         test_split: float) -> Tuple[int, int]:
        validation_index = int((1 - validation_split - test_split) * len(dataset))
        test_index = int((1 - test_split) * len(dataset))
        return test_index, validation_index

    def _equalize_sizes(self,
                        dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]],
                        adjacency_matrix_max_size: List[int],
                        features_max_size: List[int],
                        labels_max_size: List[int],
                        maximum_number_of_nodes: int,
                        maximum_number_of_features: int) -> List[Tuple[to.Tensor, to.Tensor, to.Tensor]]:
        preprocessed_dataset = []
        for features, adjacency_matrix, labels in dataset:
            if self._is_graph_size_less_than_maximum_allowed(maximum_number_of_nodes, adjacency_matrix):
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

    def _get_maximum_data_size(self,
                               dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]],
                               maximum_number_of_nodes: int) \
            -> Tuple[List, List, List]:
        features_max_size = [0, 0]
        adjacency_matrix_max_size = [0, 0]
        labels_max_size = [0]
        for features, adjacency_matrix, labels in dataset:
            if self._is_graph_size_less_than_maximum_allowed(maximum_number_of_nodes, adjacency_matrix):
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

    @staticmethod
    def _is_graph_size_less_than_maximum_allowed(maximum_number_of_nodes: int, adjacency_matrix: to.Tensor) -> bool:
        return 0 < maximum_number_of_nodes < adjacency_matrix.size()[0]

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
