import os
from unittest import TestCase

from src.repository.training_data_repository import TrainingDataRepository
from src.usecase.training import Training
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES


class TestTraining(TestCase):
    def setUp(self) -> None:
        self.training = None

    def test_start(self):
        # Given
        self.training = Training(epochs=10, loss_function='MSE', optimizer='adam')
        dataset = 'training-test-data'
        tests_path = 'tests/data'
        repository = TrainingDataRepository(dataset, tests_path)
        features = BASE_GRAPH_NODE_FEATURES
        labels = BASE_GRAPH
        filenames_to_save = ['training_features.pickle', 'training_labels.pickle']
        filenames_expected = ['tests/data/train-test-data/training_features.pickle',
                              'tests/data/train-test-data/training_labels.pickle']
        repository.save(filenames_to_save[0], features)
        repository.save(filenames_to_save[1], labels)

        # When
        running_loss = self.training.start(repository)

        # Then
        self.assertTrue(running_loss > 0.0)
        os.remove(filenames_expected[0])
        os.remove(filenames_expected[1])
