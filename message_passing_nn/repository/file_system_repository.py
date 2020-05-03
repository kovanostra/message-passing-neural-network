import logging
import os
import pickle
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
        for filename in files_in_path:
            dataset.append((self._get_features(filename), self._get_adjacency_matrix(filename), self._get_labels(filename)))
        self.get_logger().info("Finished loading dataset")
        return dataset

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
