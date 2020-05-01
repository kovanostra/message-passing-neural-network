from unittest import TestCase

import torch as to

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.model.graph import Graph
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES


class TestGraphPreprocessor(TestCase):
    def setUp(self) -> None:
        self.data_preprocessor = DataPreprocessor()

    def test_train_validation_test_split(self):
        # Given
        dataset_length = 10
        features = BASE_GRAPH_NODE_FEATURES
        labels = BASE_GRAPH
        raw_dataset = [(features, labels) for i in range(dataset_length)]
        train_validation_test_split_expected = [7, 2, 1]

        # When
        train_validation_test_split = self.data_preprocessor.train_validation_test_split(raw_dataset,
                                                                                         batch_size=1,
                                                                                         validation_split=0.2,
                                                                                         test_split=0.1)
        train_validation_test_split = [len(dataset) for dataset in train_validation_test_split]

        # Then
        self.assertEqual(train_validation_test_split_expected, train_validation_test_split)

    def test_extract_initialization_graph(self):
        # Given
        dataset_length = 1
        features = BASE_GRAPH_NODE_FEATURES
        labels = BASE_GRAPH
        raw_dataset = [(features, labels) for i in range(dataset_length)]
        initialization_graph_expected = Graph(labels, features)

        # When
        initialization_graph = self.data_preprocessor.extract_initialization_graph(raw_dataset)

        # Then
        self.assertEqual(initialization_graph_expected, initialization_graph)

    def test_flatten_when_sizes_match(self):
        # Given
        dataset_length = 2
        labels = BASE_GRAPH.view(-1)
        tensors = to.cat((labels, labels))
        tensors_flattened_expected = tensors.view(-1)

        # When
        tensors_flattened = self.data_preprocessor.flatten(tensors, desired_size=dataset_length*len(labels))

        # Then
        self.assertTrue(to.allclose(tensors_flattened_expected, tensors_flattened))

    def test_flatten_when_sizes_do_not_match(self):
        # Given
        dataset_length = 3
        labels = BASE_GRAPH.view(-1)
        tensors = to.cat((labels, labels))
        tensors_flattened_expected = to.cat((tensors.view(-1), to.zeros_like(labels)))

        # When
        tensors_flattened = self.data_preprocessor.flatten(tensors, desired_size=dataset_length*len(labels))

        # Then
        self.assertTrue(to.allclose(tensors_flattened_expected, tensors_flattened))
