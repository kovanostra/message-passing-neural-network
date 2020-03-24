from typing import Any

import torch as to
import torch.nn as nn

from src.domain.edge import Edge
from src.domain.graph import Graph
from src.domain.message_gru import Message
from src.domain.node import Node


class GraphEncoder(nn.Module):
    def __init__(self, graph: Graph, initialize_tensors=True):
        super(GraphEncoder, self).__init__()
        self.time_steps = 5
        self.w_gru_update_gate_features = None
        self.w_gru_forget_gate_features = None
        self.w_gru_current_memory_message_features = None
        self.u_gru_update_gate = None
        self.u_gru_forget_gate = None
        self.u_gru_current_memory_message = None
        self.b_gru_update_gate = None
        self.b_gru_forget_gate = None
        self.b_gru_current_memory_message = None
        self.u_graph_node_features = None
        self.u_graph_neighbor_messages = None
        if initialize_tensors:
            self._initialize_tensors(graph)

    def forward(self, graph: Graph) -> Any:
        return self.encode(graph)

    def encode(self, graph: Graph) -> Any:
        messages = self._send_messages(graph)
        encodings = self._encode_nodes(graph, messages)
        return encodings

    def _initialize_tensors(self, graph: Graph) -> None:
        base_4d_tensor_shape = [graph.number_of_nodes,
                                graph.number_of_nodes,
                                graph.number_of_node_features,
                                graph.number_of_node_features]
        base_3d_tensor_shape = [graph.number_of_nodes,
                                graph.number_of_node_features,
                                graph.number_of_node_features]
        base_2d_tensor_shape = [graph.number_of_node_features]
        self.w_gru_update_gate_features = nn.Parameter(to.rand(base_4d_tensor_shape, requires_grad=True))
        self.w_gru_forget_gate_features = nn.Parameter(to.rand(base_4d_tensor_shape, requires_grad=True))
        self.w_gru_current_memory_message_features = nn.Parameter(to.rand(base_4d_tensor_shape, requires_grad=True))
        self.u_gru_update_gate = nn.Parameter(to.rand(base_4d_tensor_shape, requires_grad=True))
        self.u_gru_forget_gate = nn.Parameter(to.rand(base_4d_tensor_shape, requires_grad=True))
        self.u_gru_current_memory_message = nn.Parameter(to.rand(base_4d_tensor_shape, requires_grad=True))
        self.b_gru_update_gate = nn.Parameter(to.rand(base_2d_tensor_shape, requires_grad=True))
        self.b_gru_forget_gate = nn.Parameter(to.rand(base_2d_tensor_shape, requires_grad=True))
        self.b_gru_current_memory_message = nn.Parameter(to.rand(base_2d_tensor_shape, requires_grad=True))
        self.u_graph_node_features = nn.Parameter(to.rand(base_3d_tensor_shape, requires_grad=True))
        self.u_graph_neighbor_messages = nn.Parameter(to.rand(base_3d_tensor_shape, requires_grad=True))

    def _send_messages(self, graph: Graph) -> Any:
        messages = to.zeros((graph.number_of_nodes,
                             graph.number_of_nodes,
                             graph.number_of_node_features))
        for step in range(self.time_steps):
            messages = self._compose_messages_from_nodes_to_targets(graph, messages)
        return messages

    def _encode_nodes(self, graph: Graph, messages: Any) -> Any:
        encoded_node = to.zeros(graph.node_features.shape)
        for node_id in range(graph.number_of_nodes):
            encoded_node[node_id] += self._apply_recurrent_layer_for_node(graph, messages, node_id)
        return encoded_node

    def _apply_recurrent_layer_for_node(self, graph: Graph, messages: Any, node_id: int) -> Any:
        node_encoding_features = self.u_graph_node_features[node_id].matmul(graph.node_features[node_id])
        node_encoding_messages = self.u_graph_neighbor_messages[node_id].matmul(to.sum(messages[node_id], dim=0))
        return to.relu(node_encoding_features + node_encoding_messages)

    def _compose_messages_from_nodes_to_targets(self, graph: Graph, messages: Any) -> Any:
        new_messages = to.zeros(messages.shape)
        for node_id in range(graph.number_of_nodes):
            node = self._create_node(graph, node_id)
            for end_node_id in node.neighbors:
                end_node = self._create_node(graph, end_node_id)
                edge = self._create_edge(graph, node, end_node)
                edge_slice = edge.get_edge_slice()
                message = self._get_message_inputs(messages, node, edge, graph)
                message.compose()
                new_messages[edge_slice] = message.value
        return new_messages

    def _get_message_inputs(self, messages: Any, node: Node, edge: Edge, graph: Graph) -> Message:
        message = self._create_message()
        message.previous_messages = self._get_messages_from_all_node_neighbors_except_target_summed(messages,
                                                                                                    node,
                                                                                                    edge).value
        message.update_gate = self._pass_through_update_gate(messages, node, edge, graph)
        message.current_memory = self._get_current_memory_message(messages, node, edge, graph)
        return message

    def _get_messages_from_all_node_neighbors_except_target_summed(self, messages: Any, node: Node,
                                                                   edge: Edge) -> Message:
        messages_from_the_other_neighbors = self._create_message()
        messages_from_the_other_neighbors.value = to.zeros(node.features.shape[0])
        if node.neighbors_count > 1:
            neighbors_slice = edge.get_start_node_neighbors_without_end_node()
            messages_from_the_other_neighbors.value = to.sum(messages[neighbors_slice], dim=0)
        return messages_from_the_other_neighbors

    def _pass_through_update_gate(self, messages: Any, node: Node, edge: Edge, graph: Graph) -> Any:
        message_from_a_neighbor_other_than_target = self._get_messages_from_all_node_neighbors_except_target_summed(
            messages,
            node,
            edge)
        edge_slice = edge.get_edge_slice()
        update_gate_output = to.sigmoid(
            self.w_gru_update_gate_features[edge_slice].matmul(graph.node_features[node.node_id]) +
            self.u_gru_update_gate[edge_slice].matmul(message_from_a_neighbor_other_than_target.value) +
            self.b_gru_update_gate)
        return update_gate_output

    def _get_current_memory_message(self, messages: Any, node: Node, edge: Edge, graph: Graph) -> Any:
        messages_passed_through_reset_gate = self._keep_or_reset_messages(messages, node, edge, graph)
        edge_slice = edge.get_edge_slice()
        current_memory_message = self.w_gru_current_memory_message_features[edge_slice].matmul(
            graph.node_features[node.node_id]) + self.u_gru_current_memory_message[edge_slice].matmul(
            messages_passed_through_reset_gate) + self.b_gru_current_memory_message
        return to.tanh(current_memory_message)

    def _keep_or_reset_messages(self, messages: Any, node: Node, edge: Edge, graph: Graph) -> Any:
        messages_from_the_other_neighbors_summed = self._create_message()
        messages_from_the_other_neighbors_summed.value = to.zeros(node.features.shape[0])
        neighbors_slice = edge.get_start_node_neighbors_without_end_node()[0]
        edge_slice = edge.get_edge_slice()
        for reset_node_index in neighbors_slice:
            reset_node = self._create_node(graph, reset_node_index)
            reset_edge = self._create_edge(graph, node, reset_node)
            reset_edge_slice = reset_edge.get_edge_slice()
            reset_gate_output = self._pass_through_reset_gate(messages, node, reset_edge, graph)
            messages_from_the_other_neighbors_summed.value += reset_gate_output * messages[reset_edge_slice]
        return self.u_gru_current_memory_message[edge_slice].matmul(messages_from_the_other_neighbors_summed.value)

    def _pass_through_reset_gate(self, messages: Any, node: Node, edge: Edge, graph: Graph) -> Any:
        edge_slice = edge.get_edge_slice()
        message_from_a_neighbor_other_than_target = messages[edge_slice]
        reset_gate_output = to.sigmoid(
            self.w_gru_update_gate_features[edge_slice].matmul(graph.node_features[node.node_id]) +
            self.u_gru_update_gate[edge_slice].matmul(message_from_a_neighbor_other_than_target) +
            self.b_gru_update_gate)
        return reset_gate_output

    @staticmethod
    def _create_node(graph: Graph, node_id: int) -> Node:
        return Node(graph, node_id)

    @staticmethod
    def _create_edge(graph: Graph, start_node: Node, end_node: Node) -> Edge:
        return Edge(graph, start_node, end_node)

    @staticmethod
    def _create_message() -> Message:
        return Message()

