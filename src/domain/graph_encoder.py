import numpy as np

from src.domain.graph import Graph
from src.domain.interface.encoder import Encoder
from src.domain.interface.messenger import Messenger


class GraphEncoder(Encoder):
    def __init__(self, messenger: Messenger):
        self.messenger = messenger
        self.u_graph_node_features = None
        self.u_graph_neighbor_messages = None

    def encode(self, graph: Graph) -> np.ndarray:
        messages = self._send_messages(graph)
        encodings = self._encode_nodes(graph, messages)
        return encodings

    def _send_messages(self, graph: Graph) -> np.ndarray:
        messages = np.zeros((graph.number_of_nodes,
                             graph.number_of_nodes,
                             graph.number_of_node_features))
        for step in range(self.messenger.time_steps):
            messages = self.messenger.compose_messages_from_nodes_to_targets(graph, messages)
        return messages

    def _encode_nodes(self, graph: Graph, messages: np.ndarray) -> np.ndarray:
        encoded_node = np.zeros(graph.node_features.shape)
        for node_id in range(graph.number_of_nodes):
            encoded_node[node_id] += self._apply_recurrent_layer_for_node(graph, messages, node_id)
        return encoded_node

    def _apply_recurrent_layer_for_node(self, graph: Graph, messages: np.ndarray, node_id: int) -> np.ndarray:
        node_encoding_features = self.u_graph_node_features[node_id].dot(graph.node_features[node_id])
        node_encoding_messages = self.u_graph_neighbor_messages[node_id].dot(np.sum(messages[node_id], axis=0))
        return self._relu(node_encoding_features + node_encoding_messages)

    @staticmethod
    def _relu(vector: np.ndarray) -> np.ndarray:
        return np.maximum(0, vector)
