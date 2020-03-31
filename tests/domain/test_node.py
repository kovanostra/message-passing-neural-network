from unittest import TestCase

import numpy as np

from src.domain.graph import Graph
from src.domain.node import Node
from tests.fixtures.matrices_and_vectors import BASE_GRAPH_NODE_FEATURES, BASE_GRAPH


class TestNode(TestCase):
    def setUp(self) -> None:
        graph = Graph(BASE_GRAPH,
                      BASE_GRAPH_NODE_FEATURES)
        self.node_id = 2
        self.node = Node(graph, self.node_id)

    def test_node_neighbors(self):
        # Given
        neighbors_expected = np.array([0, 1, 3])

        # When
        neighbors = self.node.neighbors

        # Then
        self.assertTrue(np.array_equal(neighbors_expected, neighbors))
