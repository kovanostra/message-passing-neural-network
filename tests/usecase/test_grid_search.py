import os
from typing import List
from unittest import TestCase

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.model.graph_gru_encoder import GraphGRUEncoder
from message_passing_nn.repository.file_system_repository import FileSystemRepository
from message_passing_nn.trainer.model_trainer import ModelTrainer
from message_passing_nn.usecase.grid_search import GridSearch
from message_passing_nn.utils.saver import Saver
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES


class TestTraining(TestCase):
    def setUp(self) -> None:
        self.features = BASE_GRAPH_NODE_FEATURES
        self.adjacency_matrix = BASE_GRAPH
        self.labels = BASE_GRAPH.view(-1)
        self.dataset = 'training-test-data'
        self.tests_data_directory = 'tests/test_data/'
        tests_model_directory = 'tests/model_checkpoints'
        tests_results_directory = 'tests/grid_search_results'
        device = "cpu"
        self.repository = FileSystemRepository(self.tests_data_directory, self.dataset)
        self.data_preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer(self.data_preprocessor, device)
        self.saver = Saver(tests_model_directory, tests_results_directory)

    def test_start_for_multiple_batches_of_the_same_size(self):
        # Given
        dataset_size = 6
        grid_search_dictionary = {
            "model": ["GRU"],
            "epochs": [10],
            "batch_size": [3],
            "maximum_number_of_nodes": [-1],
            "maximum_number_of_features": [-1],
            "validation_split": [0.2],
            "test_split": [0.1],
            "loss_function": ["MSE"],
            "optimizer": ["SGD"],
            "time_steps": [1],
            "validation_period": [5]
        }
        grid_search = GridSearch(self.repository,
                                 self.data_preprocessor,
                                 self.model_trainer,
                                 grid_search_dictionary,
                                 self.saver)

        features_filenames = [str(i) + '_training_features' + '.pickle' for i in range(dataset_size)]
        adjacency_matrix_filenames = [str(i) + '_training_adjacency-matrix' '.pickle' for i in range(dataset_size)]
        labels_filenames = [str(i) + '_training_labels' '.pickle' for i in range(dataset_size)]
        for i in range(dataset_size):
            self.repository.save(features_filenames[i], self.features)
            self.repository.save(adjacency_matrix_filenames[i], self.adjacency_matrix)
            self.repository.save(labels_filenames[i], self.labels)

        # When
        losses = grid_search.start()
        configuration_id = list(losses["training_loss"].keys())[0]

        # Then
        self.assertTrue(losses["training_loss"][configuration_id][grid_search_dictionary["epochs"][0]] > 0.0)
        self.assertTrue(
            losses["validation_loss"][configuration_id][grid_search_dictionary["validation_period"][0]] > 0.0)
        self.assertTrue(losses["test_loss"][configuration_id]["final_epoch"] > 0.0)

        # Tear down
        self._remove_files(dataset_size, features_filenames, adjacency_matrix_filenames, labels_filenames)

    def test_start_for_multiple_batches_of_differing_size(self):
        # Given
        dataset_size = 5
        grid_search_dictionary = {
            "model": ["RNN"],
            "epochs": [10],
            "batch_size": [3],
            "maximum_number_of_nodes": [-1],
            "maximum_number_of_features": [-1],
            "validation_split": [0.2],
            "test_split": [0.1],
            "loss_function": ["MSE"],
            "optimizer": ["SGD"],
            "time_steps": [1],
            "validation_period": [5]
        }
        grid_search = GridSearch(self.repository,
                                 self.data_preprocessor,
                                 self.model_trainer,
                                 grid_search_dictionary,
                                 self.saver)

        features_filenames = [str(i) + '_training_features' + '.pickle' for i in range(dataset_size)]
        adjacency_matrix_filenames = [str(i) + '_training_adjacency-matrix' '.pickle' for i in range(dataset_size)]
        labels_filenames = [str(i) + '_training_labels' '.pickle' for i in range(dataset_size)]
        for i in range(dataset_size):
            self.repository.save(features_filenames[i], self.features)
            self.repository.save(adjacency_matrix_filenames[i], self.adjacency_matrix)
            self.repository.save(labels_filenames[i], self.labels)

        # When
        losses = grid_search.start()
        configuration_id = list(losses["training_loss"].keys())[0]

        # Then
        self.assertTrue(losses["training_loss"][configuration_id][grid_search_dictionary["epochs"][0]] > 0.0)
        self.assertTrue(
            losses["validation_loss"][configuration_id][grid_search_dictionary["validation_period"][0]] > 0.0)
        self.assertTrue(losses["test_loss"][configuration_id]["final_epoch"] > 0.0)

        # Tear down
        self._remove_files(dataset_size, features_filenames, adjacency_matrix_filenames, labels_filenames)

    def test_start_a_grid_search(self):
        # Given
        dataset_size = 6
        grid_search_dictionary = {
            "model": ["RNN", "GRU"],
            "epochs": [10, 15],
            "batch_size": [3, 4],
            "maximum_number_of_nodes": [-1],
            "maximum_number_of_features": [-1],
            "validation_split": [0.2],
            "test_split": [0.1],
            "loss_function": ["MSE"],
            "optimizer": ["SGD"],
            "time_steps": [1],
            "validation_period": [5]
        }
        grid_search = GridSearch(self.repository,
                                 self.data_preprocessor,
                                 self.model_trainer,
                                 grid_search_dictionary,
                                 self.saver)

        features_filenames = [str(i) + '_training_features' + '.pickle' for i in range(dataset_size)]
        adjacency_matrix_filenames = [str(i) + '_training_adjacency-matrix' '.pickle' for i in range(dataset_size)]
        labels_filenames = [str(i) + '_training_labels' '.pickle' for i in range(dataset_size)]
        for i in range(dataset_size):
            self.repository.save(features_filenames[i], self.features)
            self.repository.save(adjacency_matrix_filenames[i], self.adjacency_matrix)
            self.repository.save(labels_filenames[i], self.labels)

        # When
        losses = grid_search.start()
        configuration_id = list(losses["training_loss"].keys())[0]

        # Then
        self.assertTrue(losses["training_loss"][configuration_id][grid_search_dictionary["epochs"][0]] > 0.0)
        self.assertTrue(
            losses["validation_loss"][configuration_id][grid_search_dictionary["validation_period"][0]] > 0.0)
        self.assertTrue(losses["test_loss"][configuration_id]["final_epoch"] > 0.0)

        # Tear down
        self._remove_files(dataset_size, features_filenames, adjacency_matrix_filenames, labels_filenames)

    def _remove_files(self,
                      dataset_size: int,
                      features_filenames: List[str],
                      adjacency_matrix_filenames: List[str],
                      labels_filenames: List[str]) -> None:
        for i in range(dataset_size):
            os.remove(self.tests_data_directory + self.dataset + "/" + features_filenames[i])
            os.remove(self.tests_data_directory + self.dataset + "/" + adjacency_matrix_filenames[i])
            os.remove(self.tests_data_directory + self.dataset + "/" + labels_filenames[i])
