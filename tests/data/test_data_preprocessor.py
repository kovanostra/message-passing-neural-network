from unittest import TestCase

import torch as to

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES


class TestGraphPreprocessor(TestCase):
    def setUp(self) -> None:
        self.data_preprocessor = DataPreprocessor()

    def test_train_validation_test_split(self):
        # Given
        dataset_length = 10
        features = BASE_GRAPH_NODE_FEATURES
        adjacency_matrix = BASE_GRAPH
        labels = BASE_GRAPH.view(-1)
        raw_dataset = [(features, adjacency_matrix, labels) for i in range(dataset_length)]
        train_validation_test_split_expected = [7, 2, 1]

        # When
        train_validation_test_split = self.data_preprocessor.train_validation_test_split(raw_dataset,
                                                                                         batch_size=1,
                                                                                         maximum_number_of_features=-1,
                                                                                         maximum_number_of_nodes=-1,
                                                                                         validation_split=0.2,
                                                                                         test_split=0.1)
        train_validation_test_split = [len(dataset) for dataset in train_validation_test_split]

        # Then
        self.assertEqual(train_validation_test_split_expected, train_validation_test_split)

    def test_equalize_sizes(self):
        # Given
        node_features_1 = to.ones((2, 5))
        node_features_2 = to.ones((3, 5))
        adjacency_matrix_1 = to.ones(2, 2)
        adjacency_matrix_2 = to.ones(3, 3)
        labels_1 = to.ones(15)
        labels_2 = to.ones(20)
        raw_dataset = [(node_features_1, adjacency_matrix_1, labels_1),
                       (node_features_2, adjacency_matrix_2, labels_2)]

        # When
        equalized_dataset = self.data_preprocessor.equalize_dataset_dimensions(raw_dataset, -1, -1)
        node_features_1, adjacency_matrix_1, labels_1 = equalized_dataset[0]

        # Then
        self.assertEqual(node_features_1.size(), node_features_2.size())
        self.assertEqual(adjacency_matrix_1.size(), adjacency_matrix_2.size())
        self.assertEqual(labels_1.size(), labels_2.size())

    def test_extract_data_dimensions(self):
        # Given
        dataset_length = 1
        features = BASE_GRAPH_NODE_FEATURES
        adjacency_matrix = BASE_GRAPH
        labels = BASE_GRAPH.view(-1)
        raw_dataset = [(features, adjacency_matrix, labels) for i in range(dataset_length)]
        data_dimensions_expected = (features.size(), adjacency_matrix.size(), labels.size())

        # When
        data_dimensions = self.data_preprocessor.extract_data_dimensions(raw_dataset)

        # Then
        self.assertEqual(data_dimensions_expected, data_dimensions)

    def test_flatten_when_sizes_match(self):
        # Given
        dataset_length = 2
        labels = BASE_GRAPH.view(-1)
        tensors = to.cat((labels, labels))
        tensors_flattened_expected = tensors.view(-1)

        # When
        tensors_flattened = self.data_preprocessor.flatten(tensors, desired_size=dataset_length * len(labels))

        # Then
        self.assertTrue(to.allclose(tensors_flattened_expected, tensors_flattened))

    def test_flatten_when_sizes_do_not_match(self):
        # Given
        dataset_length = 3
        labels = BASE_GRAPH.view(-1)
        tensors = to.cat((labels, labels))
        tensors_flattened_expected = to.cat((tensors.view(-1), to.zeros_like(labels)))

        # When
        tensors_flattened = self.data_preprocessor.flatten(tensors, desired_size=dataset_length * len(labels))

        # Then
        self.assertTrue(to.allclose(tensors_flattened_expected, tensors_flattened))
