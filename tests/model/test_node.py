from unittest import TestCase

import numpy as np

from message_passing_nn.model.node import Node
from tests.fixtures.matrices_and_vectors import BASE_GRAPH_NODE_FEATURES, BASE_GRAPH


class TestNode(TestCase):
    def setUp(self) -> None:
        all_node_features = BASE_GRAPH_NODE_FEATURES
        labels = BASE_GRAPH.view(-1)
        self.node = Node(all_node_features, labels, node_id=2)

    def test_node_neighbors(self):
        # Given
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
