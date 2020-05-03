from typing import Tuple, List

import torch as to


class Node:
    def __init__(self, all_node_features: to.Tensor, adjacency_matrix: to.Tensor, node_id: int):
        self.node_id = int(node_id)
        self.features = self._get_features_of_specific_node(all_node_features)
        self.symmetry = self.is_adjacency_matrix_symmetric(adjacency_matrix)
        self.all_neighbors = self._get_all_neighbors(adjacency_matrix)
        self.neighbors = self._get_neighbors(adjacency_matrix)
        self.neighbors_count = len(self.neighbors)

    @staticmethod
    def is_adjacency_matrix_symmetric(adjacency_matrix: to.Tensor) -> bool:
        return to.allclose(adjacency_matrix, adjacency_matrix.t())

    def _get_features_of_specific_node(self, all_node_features: to.Tensor) -> to.Tensor:
        return all_node_features[self.node_id]

    def _get_all_neighbors(self, adjacency_matrix: to.Tensor) -> to.Tensor:
        return to.nonzero(adjacency_matrix[self.node_id], as_tuple=True)[0]

    def _get_neighbors(self, adjacency_matrix: to.Tensor) -> List:
        neighbors = self._get_all_neighbors(adjacency_matrix)
        if self.symmetry:
            neighbors = self._get_greater_neighbors(neighbors)
        return neighbors

    def _get_greater_neighbors(self, neighbors: List) -> List:
        return [neighbor for neighbor in neighbors if neighbor > self.node_id]

    def get_start_node_neighbors_without_end_node(self, end_node_id: int) -> Tuple:
        return self._remove_end_node_from_start_node_neighbors(end_node_id), [self.node_id]

    def _remove_end_node_from_start_node_neighbors(self, end_node_id: int) -> to.tensor:
        end_node_index = (self.all_neighbors == end_node_id).nonzero()[0][0].item()
        return to.cat((self.all_neighbors[:end_node_index],
                       self.all_neighbors[end_node_index + 1:])).tolist()