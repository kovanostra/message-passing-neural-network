from typing import Tuple, Any

import torch as to

from message_passing_nn.model.node import Node


class Edge:
    def __init__(self, start_node: Node, end_node: Node) -> None:
        self.start_node = start_node
        self.end_node = end_node

    def get_edge_slice(self) -> Tuple:
        return self.start_node.node_id, self._get_integer(self.end_node.node_id)

    def get_start_node_neighbors_without_end_node(self) -> Tuple:
        return self._remove_end_node_from_start_node_neighbors(), [self.start_node.node_id]

    def _remove_end_node_from_start_node_neighbors(self) -> to.tensor:
        end_node_index = (self.start_node.neighbors == self.end_node.node_id).nonzero()[0][0].item()
        return to.cat((self.start_node.neighbors[:end_node_index],
                       self.start_node.neighbors[end_node_index + 1:])).tolist()

    @staticmethod
    def _get_integer(field: Any) -> int:
        return int(field)
