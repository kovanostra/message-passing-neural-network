from typing import Any

import torch as to

from src.domain.graph import Graph


class Node:
    def __init__(self, graph: Graph, node_id: int):
        self.node_id = node_id
        self.features = self._get_node_features(graph)
        self.neighbors = self._get_neighbors(graph)
        self.neighbors_count = len(self.neighbors)

    def _get_node_features(self, graph: Graph) -> Any:
        return graph.node_features[self.node_id]

    def _get_neighbors(self, graph: Graph) -> Any:
        return to.nonzero(graph.adjacency_matrix[self.node_id], as_tuple=True)[0]
