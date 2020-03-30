from unittest import TestCase

from src.domain.graph import Graph
from src.domain.graph_preprocessor import GraphPreprocessor
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES


class TestGraphPreprocessor(TestCase):
    def setUp(self) -> None:
        self.graph_preprocessor = GraphPreprocessor()

    def test_preprocess_returns_graph_objects(self):
        # Given
        features = BASE_GRAPH_NODE_FEATURES
        labels = BASE_GRAPH
        dataset_unprocessed = [(features, labels)]
        graph_expected = Graph(BASE_GRAPH, BASE_GRAPH_NODE_FEATURES)
        batches = 1
        dataset_expected = [[graph_expected]]

        # When
        dataset = self.graph_preprocessor.preprocess(dataset_unprocessed, batches)

        # Then
        self.assertEqual(dataset_expected, dataset)

    def test_preprocess_returns_dataset_in_batches(self):
        # Given
        features = BASE_GRAPH_NODE_FEATURES
        labels = BASE_GRAPH
        dataset_length = 95
        batch_length_expected = [32, 32, 31]
        dataset_unprocessed = [(features, labels) for i in range(dataset_length)]
        batches = 3

        # When
        dataset_in_batches = self.graph_preprocessor.preprocess(dataset_unprocessed, batches)

        # Then
        self.assertEqual(len(dataset_in_batches[0]), batch_length_expected[0])
        self.assertEqual(len(dataset_in_batches[1]), batch_length_expected[1])
        self.assertEqual(len(dataset_in_batches[2]), batch_length_expected[2])

