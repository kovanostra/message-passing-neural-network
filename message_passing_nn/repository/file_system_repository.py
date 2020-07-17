import logging
import os
import pickle
import sys
from typing import List, Tuple

import torch as to

from message_passing_nn.repository.repository import Repository


class FileSystemRepository(Repository):
    def __init__(self, data_directory: str, dataset: str) -> None:
        super().__init__()
        self.data_directory = data_directory + dataset + '/'

    def save(self, filename: str, data_to_save: to.Tensor) -> None:
        with open(self.data_directory + filename, 'wb') as file:
            pickle.dump(data_to_save, file)

    def get_all_data(self) -> List[Tuple[to.Tensor, to.Tensor, to.Tensor]]:
        self.get_logger().info("Loading dataset")
        files_in_path = self._extract_name_prefixes_from_filenames()
        dataset = []
        size = 0
        for filename in files_in_path:
            dataset.append(
                (self._get_features(filename), self._get_adjacency_matrix(filename), self._get_labels(filename)))
            size += self._get_size(dataset[-1])
        self.get_logger().info(
            "Loaded " + str(len(dataset)) + " files. Size: " + str(int(size * 0.000001)) + " MB")
        return dataset

    @staticmethod
    def _get_size(data: Tuple[to.Tensor, to.Tensor, to.Tensor]) -> int:
        return int(data[0].element_size() * data[0].nelement() +
                   data[1].element_size() * data[1].nelement() +
                   data[2].element_size() * data[2].nelement())

    def _get_labels(self, filename: str) -> to.Tensor:
        with open(self.data_directory + filename + 'labels.pickle', 'rb') as labels_file:
            labels = pickle.load(labels_file).float()
        return labels

    def _get_features(self, filename: str) -> to.Tensor:
        with open(self.data_directory + filename + 'features.pickle', 'rb') as features_file:
            features = pickle.load(features_file).float()
        return features

    def _get_adjacency_matrix(self, filename: str) -> to.Tensor:
        with open(self.data_directory + filename + 'adjacency-matrix.pickle', 'rb') as adjacency_matrix_file:
            features = pickle.load(adjacency_matrix_file).float()
        return features

    def _extract_name_prefixes_from_filenames(self):
        return set([self._reconstruct_filename(file) for file in self._get_data_filenames()])

    def _get_data_filenames(self):
        return sorted([file for file in os.listdir(self.data_directory) if file.endswith(".pickle")])

    @staticmethod
    def _reconstruct_filename(file):
        return "_".join(file.split("_")[:-1]) + "_"

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
