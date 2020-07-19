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
                                    dataset: List[Tuple[to.Tensor, List[List[int]], to.Tensor]],
                                    batch_size: int,
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
        self.get_logger().info("Train/validation/test split: " + "/".join([str(len(training_data)),
                                                                           str(len(validation_data)),
                                                                           str(len(test_data))])
                               + " batches of " + str(batch_size))
        return training_data, validation_data, test_data

    @staticmethod
    def get_dataloader(dataset: List[Tuple[to.Tensor, List[List[int]], to.Tensor]], batch_size: int = 1) -> DataLoader:
        return DataLoader(GraphDataset(dataset), batch_size)

    @staticmethod
    def find_all_node_neighbors(dataset: List[Tuple[to.Tensor, to.Tensor, to.Tensor]]) -> List[
        Tuple[to.Tensor, List[List[int]], to.Tensor]]:
        dataset_with_neighbors = []
        for element in dataset:
            features, adjacency_matrix, labels = element
            all_neighbors = [to.nonzero(adjacency_matrix[node_id], as_tuple=True)[0].tolist() for node_id in
                             range(adjacency_matrix.shape[0])]
            dataset_with_neighbors.append((features, all_neighbors, labels))
        return dataset_with_neighbors

    @staticmethod
    def extract_data_dimensions(dataset: List[Tuple[to.Tensor, List[List[int]], to.Tensor]]) -> Tuple:
        node_features_size = dataset[0][0].size()
        labels_size = dataset[0][2].size()
        return node_features_size, labels_size

    @staticmethod
    def flatten(tensors: to.Tensor, desired_size: int = 0) -> to.Tensor:
        flattened_tensor = tensors.view(-1)
        if 0 < desired_size != len(flattened_tensor):
            flattened_tensor = DataPreprocessor._pad_zeros(flattened_tensor, desired_size)
        return flattened_tensor

    @staticmethod
    def normalize(tensor: to.Tensor, device: str) -> to.Tensor:
        if tensor.size()[0] > 1:
            normalizer = nn.BatchNorm1d(tensor.size()[1], affine=False).to(device)
            return normalizer(tensor)
        else:
            return tensor

    @staticmethod
    def _pad_zeros(flattened_tensor: to.Tensor, desired_size: int) -> to.Tensor:
        size_difference = abs(len(flattened_tensor) - desired_size)
        flattened_tensor = to.cat((flattened_tensor, to.zeros(size_difference)))
        return flattened_tensor

    @staticmethod
    def _get_validation_and_test_indexes(dataset: List[Tuple[to.Tensor, List[List[int]], to.Tensor]],
                                         validation_split: float,
                                         test_split: float) -> Tuple[int, int]:
        validation_index = int((1 - validation_split - test_split) * len(dataset))
        test_index = int((1 - test_split) * len(dataset))
        return test_index, validation_index

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
