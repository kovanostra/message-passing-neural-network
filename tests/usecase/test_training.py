import os
from unittest import TestCase

import torch as to
from torch import nn

from src.repository.training_data_repository import TrainingDataRepository
from src.usecase.training import Training
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES


class TestTraining(TestCase):
    def test_start_for_multiple_batches_of_the_same_size(self):
        # Given
        batch_size = 3
        dataset_size = 6
        loss_function = nn.MSELoss()
        optimizer = to.optim.SGD
        dataset = 'training-test-data'
        tests_data_path = 'tests/data/'
        repository = TrainingDataRepository(tests_data_path, dataset)
        training = Training(repository, epochs=10, loss_function=loss_function, optimizer=optimizer)

        features = BASE_GRAPH_NODE_FEATURES
        labels = BASE_GRAPH
        features_filenames = [str(i) + '_training_features' + '.pickle' for i in range(dataset_size)]
        labels_filenames = [str(i) + '_training_labels' '.pickle' for i in range(dataset_size)]
        for i in range(dataset_size):
            repository.save(features_filenames[i], features)
            repository.save(labels_filenames[i], labels)

        # When
        training.start(batch_size)

        # Then
        self.assertTrue(training.running_loss > 0.0)
        for i in range(dataset_size):
            os.remove(tests_data_path + dataset + "/" + features_filenames[i])
            os.remove(tests_data_path + dataset + "/" + labels_filenames[i])

    def test_start_for_multiple_batches_of_differing_size(self):
        # Given
        batch_size = 3
        dataset_size = 5
        loss_function = nn.MSELoss()
        optimizer = to.optim.SGD
        dataset = 'training-test-data'
        tests_data_path = 'tests/data/'
        repository = TrainingDataRepository(tests_data_path, dataset)
        training = Training(repository, epochs=10, loss_function=loss_function, optimizer=optimizer)

        features = BASE_GRAPH_NODE_FEATURES
        labels = BASE_GRAPH
        features_filenames = [str(i) + '_training_features' + '.pickle' for i in range(dataset_size)]
        labels_filenames = [str(i) + '_training_labels' '.pickle' for i in range(dataset_size)]
        for i in range(dataset_size):
            repository.save(features_filenames[i], features)
            repository.save(labels_filenames[i], labels)

        # When
        training.start(batch_size)

        # Then
        self.assertTrue(training.running_loss > 0.0)
        for i in range(dataset_size):
            os.remove(tests_data_path + dataset + "/" + features_filenames[i])
            os.remove(tests_data_path + dataset + "/" + labels_filenames[i])
