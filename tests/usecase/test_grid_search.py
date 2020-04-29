import os
from unittest import TestCase

from src.domain.graph_encoder import GraphEncoder
from src.domain.model_trainer import ModelTrainer
from src.domain.saver import Saver
from src.repository.training_data_repository import TrainingDataRepository
from src.usecase.grid_search import GridSearch
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES


class TestTraining(TestCase):
    def test_start_for_multiple_batches_of_the_same_size(self):
        # Given
        dataset_size = 6
        grid_search_dictionary = {
            "epochs": [10],
            "batch_size": [3],
            "validation_split": [0.2],
            "test_split": [0.1],
            "loss_function": ["MSE"],
            "optimizer": ["SGD"],
            "time_steps": [1],
            "validation_period": [5]
        }
        dataset = 'training-test-data'
        tests_data_directory = 'tests/data/'
        tests_model_directory = 'tests/model'
        tests_results_directory = 'tests/results'
        repository = TrainingDataRepository(tests_data_directory, dataset)
        model_trainer = ModelTrainer(GraphEncoder)
        saver = Saver(tests_model_directory, tests_results_directory)
        grid_search = GridSearch(repository, model_trainer, grid_search_dictionary, saver)

        features = BASE_GRAPH_NODE_FEATURES
        labels = BASE_GRAPH
        features_filenames = [str(i) + '_training_features' + '.pickle' for i in range(dataset_size)]
        labels_filenames = [str(i) + '_training_labels' '.pickle' for i in range(dataset_size)]
        for i in range(dataset_size):
            repository.save(features_filenames[i], features)
            repository.save(labels_filenames[i], labels)

        # When
        losses = grid_search.start()

        # Then
        configuration_id = list(losses["training_loss"].keys())[0]
        self.assertTrue(losses["training_loss"][configuration_id][grid_search_dictionary["epochs"][0]] > 0.0)
        self.assertTrue(losses["validation_loss"][configuration_id][grid_search_dictionary["validation_period"][0]] > 0.0)
        self.assertTrue(losses["test_loss"][configuration_id]["final_epoch"] > 0.0)
        for i in range(dataset_size):
            os.remove(tests_data_directory + dataset + "/" + features_filenames[i])
            os.remove(tests_data_directory + dataset + "/" + labels_filenames[i])

    def test_start_for_multiple_batches_of_differing_size(self):
        # Given
        dataset_size = 5
        grid_search_dictionary = {
            "epochs": [10],
            "batch_size": [3],
            "validation_split": [0.2],
            "test_split": [0.1],
            "loss_function": ["MSE"],
            "optimizer": ["SGD"],
            "time_steps": [1],
            "validation_period": [5]
        }
        dataset = 'training-test-data'
        tests_data_directory = 'tests/data/'
        tests_model_directory = 'tests/model'
        tests_results_directory = 'tests/results'
        repository = TrainingDataRepository(tests_data_directory, dataset)
        model_trainer = ModelTrainer(GraphEncoder)
        saver = Saver(tests_model_directory, tests_results_directory)
        grid_search = GridSearch(repository, model_trainer, grid_search_dictionary, saver)

        features = BASE_GRAPH_NODE_FEATURES
        labels = BASE_GRAPH
        features_filenames = [str(i) + '_training_features' + '.pickle' for i in range(dataset_size)]
        labels_filenames = [str(i) + '_training_labels' '.pickle' for i in range(dataset_size)]
        for i in range(dataset_size):
            repository.save(features_filenames[i], features)
            repository.save(labels_filenames[i], labels)

        # When
        losses = grid_search.start()

        # Then
        configuration_id = list(losses["training_loss"].keys())[0]
        self.assertTrue(losses["training_loss"][configuration_id][grid_search_dictionary["epochs"][0]] > 0.0)
        self.assertTrue(losses["validation_loss"][configuration_id][grid_search_dictionary["validation_period"][0]] > 0.0)
        self.assertTrue(losses["test_loss"][configuration_id]["final_epoch"] > 0.0)
        for i in range(dataset_size):
            os.remove(tests_data_directory + dataset + "/" + features_filenames[i])
            os.remove(tests_data_directory + dataset + "/" + labels_filenames[i])

    def test_start_a_grid_search(self):
        # Given
        dataset_size = 5
        grid_search_dictionary = {
            "epochs": [10, 15],
            "batch_size": [3, 4],
            "validation_split": [0.2],
            "test_split": [0.1],
            "loss_function": ["MSE"],
            "optimizer": ["SGD"],
            "time_steps": [1],
            "validation_period": [5]
        }
        dataset = 'training-test-data'
        tests_data_directory = 'tests/data/'
        tests_model_directory = 'tests/model'
        tests_results_directory = 'tests/results'
        repository = TrainingDataRepository(tests_data_directory, dataset)
        model_trainer = ModelTrainer(GraphEncoder)
        saver = Saver(tests_model_directory, tests_results_directory)
        grid_search = GridSearch(repository, model_trainer, grid_search_dictionary, saver)

        features = BASE_GRAPH_NODE_FEATURES
        labels = BASE_GRAPH
        features_filenames = [str(i) + '_training_features' + '.pickle' for i in range(dataset_size)]
        labels_filenames = [str(i) + '_training_labels' '.pickle' for i in range(dataset_size)]
        for i in range(dataset_size):
            repository.save(features_filenames[i], features)
            repository.save(labels_filenames[i], labels)

        # When
        losses = grid_search.start()

        # Then
        configuration_id = list(losses["training_loss"].keys())[0]
        self.assertTrue(losses["training_loss"][configuration_id][grid_search_dictionary["epochs"][0]] > 0.0)
        self.assertTrue(losses["validation_loss"][configuration_id][grid_search_dictionary["validation_period"][0]] > 0.0)
        self.assertTrue(losses["test_loss"][configuration_id]["final_epoch"] > 0.0)
        for i in range(dataset_size):
            os.remove(tests_data_directory + dataset + "/" + features_filenames[i])
            os.remove(tests_data_directory + dataset + "/" + labels_filenames[i])

