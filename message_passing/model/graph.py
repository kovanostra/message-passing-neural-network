import torch as to


class Graph:
    def __init__(self, adjacency_matrix: to.Tensor, node_features: to.Tensor) -> None:
        self.adjacency_matrix = adjacency_matrix
        self.node_features = node_features
        self.number_of_nodes = self._get_number_of_nodes()
        self.number_of_node_features = self._get_number_of_node_features()

    def __eq__(self, o):
        return to.all(to.eq(self.adjacency_matrix, o.adjacency_matrix)) and \
               to.all(to.eq(self.node_features, o.node_features))

    def _get_number_of_nodes(self) -> int:
        return self.adjacency_matrix.shape[0]

    def _get_number_of_node_features(self) -> int:
        return self.node_features.shape[1]
