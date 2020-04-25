from typing import Any

import numpy as np
import torch as to


class Node:
    def __init__(self, all_node_features: Any, adjacency_matrix: Any, node_id: Any):
        self.node_id = self._get_integer(node_id)
        self.features = self._get_features_of_specific_node(all_node_features)
        self.neighbors = self._get_neighbors(adjacency_matrix)
        self.neighbors_count = len(self.neighbors)

    def _get_features_of_specific_node(self, all_node_features: Any) -> Any:
        return all_node_features[self.node_id]

    def _get_neighbors(self, adjacency_matrix: Any) -> Any:
        number_of_nodes = self._get_integer(np.sqrt(adjacency_matrix.size()[-1]))
        adjacency_matrix = adjacency_matrix.view(number_of_nodes, number_of_nodes)
        return to.nonzero(adjacency_matrix[self.node_id], as_tuple=True)[0]

    @staticmethod
    def _get_integer(field: Any) -> int:
        return int(field)

