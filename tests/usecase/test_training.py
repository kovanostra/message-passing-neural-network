import os
from unittest import TestCase

import torch as to
from torch import nn

from src.repository.training_data_repository import TrainingDataRepository
from src.usecase.training import Training
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES


class TestTraining(TestCase):
    def test_start(self):
        # Given
        loss_function = nn.MSELoss()
        optimizer = to.optim.SGD
        training = Training(epochs=10, loss_function=loss_function, optimizer=optimizer)
        device = "cpu"
        multiple_gpus = False
        dataset = 'training-test-data'
        tests_data_path = 'tests/data/'
        repository = TrainingDataRepository(tests_data_path, dataset)
        features = BASE_GRAPH_NODE_FEATURES
        labels = BASE_GRAPH
        filenames_to_save = ['training_features.pickle', 'training_labels.pickle']
        filenames_expected = ['tests/data/training-test-data/training_features.pickle',
                              'tests/data/training-test-data/training_labels.pickle']
        repository.save(filenames_to_save[0], features)
        repository.save(filenames_to_save[1], labels)

        # When
        training.start(repository, device, multiple_gpus)

        # Then
        self.assertTrue(training.running_loss > 0.0)
        os.remove(filenames_expected[0])
        os.remove(filenames_expected[1])
