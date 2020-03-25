import os
from unittest import TestCase

from src.domain.graph import Graph
from src.repository.training_data_repository import TrainingDataRepository
from src.usecase.train import Train
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, BASE_GRAPH_EDGE_FEATURES


class TestTrain(TestCase):
    def setUp(self) -> None:
        self.train = None

    def test_start(self):
        # Given
        self.train = Train(epochs=10, loss_function='MSE', optimizer='adam')
        tests_path = 'tests/data/'
        filename = 'dataset.pickle'
        repository = TrainingDataRepository(tests_path)
        graph_expected = [Graph(BASE_GRAPH,
                                BASE_GRAPH_NODE_FEATURES,
                                BASE_GRAPH_EDGE_FEATURES)]
        repository.save(graph_expected)

        # When
        running_loss = self.train.start(repository)

        # Then
        self.assertTrue(running_loss > 0.0)
        os.remove(tests_path + filename)
