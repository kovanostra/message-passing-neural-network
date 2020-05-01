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
