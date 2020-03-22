from typing import Tuple

import numpy as np

from src.domain.graph import Graph
from src.domain.node import Node


class Edge:
    def __init__(self, graph: Graph, start_node: Node, end_node: Node) -> None:
        self.start_node = start_node
        self.end_node = end_node
        self.features = self._extract_edge_features(graph)

    def get_edge_slice(self) -> Tuple:
        return self.start_node.node_id, self.end_node.node_id

    def get_start_node_neighbors_without_end_node(self) -> Tuple:
        return self._remove_end_node_from_start_node_neighbors(), self.start_node.node_id

    def _extract_edge_features(self, graph: Graph) -> np.ndarray:
        return graph.edge_features[self.start_node.node_id, self.end_node.node_id]

    def _remove_end_node_from_start_node_neighbors(self) -> np.ndarray:
        end_node_index = np.argwhere(self.start_node.neighbors == self.end_node.node_id)
        return np.delete(self.start_node.neighbors, end_node_index)
