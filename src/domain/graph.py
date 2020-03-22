import numpy as np


class Graph:
    def __init__(self, adjacency_matrix: np.ndarray, node_features: np.ndarray, edge_features: np.ndarray) -> None:
        self.adjacency_matrix = adjacency_matrix
        self.node_features = node_features
        self.edge_features = edge_features
        self.number_of_nodes = self._get_number_of_nodes()
        self.number_of_node_features = self._get_number_of_node_features()

    def __eq__(self, o):
        return np.array_equal(self.adjacency_matrix, o.adjacency_matrix) and \
               np.array_equal(self.node_features, o.node_features) and \
               np.array_equal(self.edge_features, o.edge_features)

    def _get_number_of_nodes(self):
        return self.adjacency_matrix.shape[0]

    def _get_number_of_node_features(self):
        return self.node_features.shape[1]
