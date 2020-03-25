import os.path
from os import path
from unittest import TestCase

from src.domain.graph import Graph
from src.repository.training_data_repository import TrainingDataRepository
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, BASE_GRAPH_EDGE_FEATURES


class TestTrainingDataRepository(TestCase):
    def setUp(self) -> None:
        tests_path = 'tests/data/'
        self.training_data_repository = TrainingDataRepository(tests_path)

    def test_save(self):
        # Given
        graph = [Graph(BASE_GRAPH,
                       BASE_GRAPH_NODE_FEATURES,
                       BASE_GRAPH_EDGE_FEATURES)]

        filename_expected = 'tests/data/dataset.pickle'

        # When
        self.training_data_repository.save(graph)

        # Then
        path.exists(filename_expected)
        os.remove(filename_expected)

    def test_get_all(self):
        # Given
        graph_expected = [Graph(BASE_GRAPH,
                                BASE_GRAPH_NODE_FEATURES,
                                BASE_GRAPH_EDGE_FEATURES)]
        self.training_data_repository.save(graph_expected)

        # When
        dataset = self.training_data_repository.get_all()

        # Then
        self.assertTrue(graph_expected[0] == dataset[0])
