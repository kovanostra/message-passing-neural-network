from typing import Any

import torch as to


class Graph:
    def __init__(self, adjacency_matrix: Any, node_features: Any) -> None:
        self.adjacency_matrix = adjacency_matrix
        self.node_features = node_features
        self.number_of_nodes = self._get_number_of_nodes()
        self.number_of_node_features = self._get_number_of_node_features()

    def __eq__(self, o):
        return to.all(to.eq(self.adjacency_matrix, o.adjacency_matrix)) and \
               to.all(to.eq(self.node_features, o.node_features))

    def _get_number_of_nodes(self):
        return self.adjacency_matrix.shape[0]

    def _get_number_of_node_features(self):
        return self.node_features.shape[1]
