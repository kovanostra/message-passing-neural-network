from unittest import TestCase

import numpy as np

from message_passing_nn.model.node import Node
from tests.fixtures.matrices_and_vectors import BASE_GRAPH_NODE_FEATURES, BASE_GRAPH, NOT_SYMMETRIC_GRAPH


class TestNode(TestCase):
    def setUp(self) -> None:
        all_node_features = BASE_GRAPH_NODE_FEATURES
        labels = BASE_GRAPH
        self.node = Node(all_node_features, labels, node_id=2)

    def test_is_matrix_symmetric(self):
        # Given
        symmetric_matrix = BASE_GRAPH

        # When
        symmetry = Node.is_adjacency_matrix_symmetric(symmetric_matrix)

        # Then
        self.assertTrue(symmetry)

    def test_is_matrix_not_symmetric(self):
        # Given
        not_symmetric_matrix = NOT_SYMMETRIC_GRAPH

        # When
        symmetry = Node.is_adjacency_matrix_symmetric(not_symmetric_matrix)

        # Then
        self.assertFalse(symmetry)

    def test_node_neighbors_symmetric(self):
        # Given
        neighbors_expected = np.array([3])

        # When
        neighbors = self.node.neighbors

        # Then
        self.assertTrue(np.array_equal(neighbors_expected, neighbors))

    def test_node_neighbors_not_symmetric(self):
        # Given
        all_node_features = BASE_GRAPH_NODE_FEATURES
        labels = NOT_SYMMETRIC_GRAPH
        self.node = Node(all_node_features, labels, node_id=2)
        neighbors_expected = np.array([0, 1, 3])

        # When
        neighbors = self.node.neighbors

        # Then
        self.assertTrue(np.array_equal(neighbors_expected, neighbors))

    def test_get_start_node_neighbors_without_end_node(self):
        # Given
        start_node_neighbors_expected = np.array([1, 3])
        end_node_id = 0

        # When
        start_node_neighbors = self.node.get_start_node_neighbors_without_end_node(end_node_id)[0]

        # Then
        self.assertTrue(np.array_equal(start_node_neighbors_expected, start_node_neighbors))
