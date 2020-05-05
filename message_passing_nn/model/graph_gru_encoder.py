from typing import List

import torch as to
import torch.nn as nn

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.model.node import Node


class GraphGRUEncoder(nn.Module):
    def __init__(self,
                 time_steps: int,
                 number_of_nodes: int,
                 number_of_node_features: int,
                 fully_connected_layer_input_size: int,
                 fully_connected_layer_output_size: int,
                 device: str) -> None:
        super(GraphGRUEncoder, self).__init__()
        base_4d_tensor_shape = [number_of_nodes, number_of_nodes, number_of_node_features, number_of_node_features]
        base_3d_tensor_shape = [number_of_nodes, number_of_node_features, number_of_node_features]
        base_2d_tensor_shape = [number_of_node_features, 1]

        self.time_steps = time_steps
        self.number_of_nodes = number_of_nodes
        self.number_of_node_features = number_of_node_features
        self.fully_connected_layer_input_size = fully_connected_layer_input_size
        self.fully_connected_layer_output_size = fully_connected_layer_output_size
        self.device = device

        self.w_gru_update_gate_features = self._get_parameter(base_4d_tensor_shape)
        self.w_gru_forget_gate_features = self._get_parameter(base_4d_tensor_shape)
        self.w_gru_current_memory_message_features = self._get_parameter(base_4d_tensor_shape)
        self.u_gru_update_gate = self._get_parameter(base_4d_tensor_shape)
        self.u_gru_forget_gate = self._get_parameter(base_4d_tensor_shape)
        self.u_gru_current_memory_message = self._get_parameter(base_4d_tensor_shape)
        self.b_gru_update_gate = self._get_parameter(base_2d_tensor_shape).view(base_2d_tensor_shape[0])
        self.b_gru_forget_gate = self._get_parameter(base_2d_tensor_shape).view(base_2d_tensor_shape[0])
        self.b_gru_current_memory_message = self._get_parameter(base_2d_tensor_shape).view(base_2d_tensor_shape[0])
        self.u_graph_node_features = self._get_parameter(base_3d_tensor_shape)
        self.u_graph_neighbor_messages = self._get_parameter(base_3d_tensor_shape)
        self.linear = to.nn.Linear(self.fully_connected_layer_input_size, self.fully_connected_layer_output_size)
        self.sigmoid = to.nn.Sigmoid()

    @classmethod
    def of(cls,
           time_steps: int,
           number_of_nodes: int,
           number_of_node_features: int,
           fully_connected_layer_input_size: int,
           fully_connected_layer_output_size: int,
           device: str):
        return cls(time_steps,
                   number_of_nodes,
                   number_of_node_features,
                   fully_connected_layer_input_size,
                   fully_connected_layer_output_size,
                   device)

    def forward(self,
                node_features: to.Tensor,
                adjacency_matrix: to.Tensor,
                batch_size: int) -> to.Tensor:
        outputs = to.zeros(batch_size, self.fully_connected_layer_output_size, device=self.device)
        for batch in range(batch_size):
            outputs[batch] = self.sigmoid(
                self.linear(
                    DataPreprocessor.flatten(
                        self.encode(node_features[batch], adjacency_matrix[batch]),
                        self.fully_connected_layer_input_size)))
        return outputs

    def encode(self, node_features: to.Tensor, adjacency_matrix: to.Tensor) -> to.Tensor:
        messages = self._send_messages(node_features, adjacency_matrix)
        encodings = self._encode_nodes(node_features, messages)
        return encodings

    def _send_messages(self, node_features: to.Tensor, adjacency_matrix: to.Tensor) -> to.Tensor:
        messages = to.zeros((self.number_of_nodes, self.number_of_nodes, self.number_of_node_features),
                            device=self.device)
        for step in range(self.time_steps):
            messages = self._compose_messages(node_features, adjacency_matrix, messages)
        return messages

    def _encode_nodes(self, node_features: to.Tensor, messages: to.Tensor) -> to.Tensor:
        encoded_node = to.zeros(self.number_of_nodes, self.number_of_node_features, device=self.device)
        for node_id in range(self.number_of_nodes):
            encoded_node[node_id] = self._apply_recurrent_layer(node_features, messages, node_id)
        return encoded_node

    def _apply_recurrent_layer(self, node_features: to.Tensor, messages: to.Tensor, node_id: int) -> to.Tensor:
        node_encoding_features = self.u_graph_node_features[node_id].matmul(node_features[node_id])
        node_encoding_messages = self.u_graph_neighbor_messages[node_id].matmul(to.sum(messages[node_id], dim=0))
        return to.relu(node_encoding_features + node_encoding_messages)

    def _compose_messages(self,
                          node_features: to.Tensor,
                          adjacency_matrix: to.Tensor,
                          messages: to.Tensor) -> to.Tensor:
        new_messages = to.zeros(messages.shape, device=self.device)
        for node_id in range(self.number_of_nodes):
            node = self._create_node(node_features, adjacency_matrix, node_id)
            for end_node_id in node.neighbors:
                message = self._get_message_value(messages, node, end_node_id, node_features)
                new_messages[node_id, end_node_id] = message
                if Node.is_adjacency_matrix_symmetric(adjacency_matrix):
                    new_messages[end_node_id, node_id] = message
        return new_messages

    def _get_message_value(self,
                           messages: to.Tensor,
                           node: Node,
                           end_node_id: int,
                           node_features: to.Tensor) -> to.Tensor:
        previous_messages = self._sum_messages_from_neighbors_except_target(messages,
                                                                            node,
                                                                            end_node_id)
        update_gate = self._pass_through_update_gate(messages, node, end_node_id, node_features)
        current_memory = self._get_current_memory_message(messages, node, end_node_id, node_features)
        return to.add(
            to.mul(
                to.sub(to.ones(update_gate.shape, device=self.device),
                       update_gate),
                previous_messages),
            to.mul(update_gate,
                   current_memory))

    def _sum_messages_from_neighbors_except_target(self,
                                                   messages: to.Tensor,
                                                   node: Node,
                                                   end_node_id: int) -> to.Tensor:
        messages_from_the_other_neighbors = to.zeros(node.features.shape[0], device=self.device)
        if node.neighbors_count > 1:
            neighbors_slice = node.get_start_node_neighbors_without_end_node(end_node_id)
            messages_from_the_other_neighbors = to.sum(messages[neighbors_slice], dim=0)
        return messages_from_the_other_neighbors

    def _pass_through_update_gate(self,
                                  messages: to.Tensor,
                                  node: Node,
                                  end_node_id: int,
                                  node_features: to.Tensor) -> to.Tensor:
        message_from_a_neighbor_other_than_target = self._sum_messages_from_neighbors_except_target(
            messages,
            node,
            end_node_id)
        update_gate_output = to.sigmoid(
            to.add(
                to.add(self.w_gru_update_gate_features[node.node_id, end_node_id].matmul(node_features[node.node_id]),
                       self.u_gru_update_gate[node.node_id, end_node_id].matmul(
                           message_from_a_neighbor_other_than_target)),
                self.b_gru_update_gate))
        return update_gate_output

    def _get_current_memory_message(self,
                                    messages: to.Tensor,
                                    node: Node,
                                    end_node_id: int,
                                    node_features: to.Tensor) -> to.Tensor:
        messages_passed_through_reset_gate = self._keep_or_reset_messages(messages,
                                                                          node,
                                                                          end_node_id,
                                                                          node_features)
        current_memory_message = to.add(
            to.add(self.w_gru_current_memory_message_features[node.node_id, end_node_id].matmul(
                node_features[node.node_id]),
                   self.u_gru_current_memory_message[node.node_id, end_node_id].matmul(
                       messages_passed_through_reset_gate)),
            self.b_gru_current_memory_message)
        return to.tanh(current_memory_message)

    def _keep_or_reset_messages(self,
                                messages: to.Tensor,
                                node: Node,
                                end_node_id: int,
                                node_features: to.Tensor) -> to.Tensor:
        return self.u_gru_current_memory_message[node.node_id, end_node_id].matmul(sum([to.mul(to.sigmoid(
            to.add(
                to.add(self.w_gru_update_gate_features[node.node_id, end_node_id].matmul(node_features[node.node_id]),
                       self.u_gru_update_gate[node.node_id, end_node_id].matmul(
                           messages[node.node_id, reset_node_index])),
                self.b_gru_update_gate)).long(), messages[node.node_id, reset_node_index])
                                                                                        for reset_node_index in
                                                                                        node.get_start_node_neighbors_without_end_node(
                                                                                            end_node_id)[0]]))

    def _pass_through_reset_gate(self,
                                 messages: to.Tensor,
                                 node: Node,
                                 end_node_id: int,
                                 node_features: to.Tensor) -> to.Tensor:
        return to.sigmoid(
            to.add(
                to.add(self.w_gru_update_gate_features[node.node_id, end_node_id].matmul(node_features[node.node_id]),
                       self.u_gru_update_gate[node.node_id, end_node_id].matmul(messages[node.node_id, end_node_id])),
                self.b_gru_update_gate)).long()

    def _get_parameter(self, tensor_shape: List[int]) -> nn.Parameter:
        return nn.Parameter(nn.init.kaiming_normal_(to.zeros(tensor_shape, device=self.device)), requires_grad=True)

    @staticmethod
    def _create_node(node_features: to.Tensor, adjacency_matrix: to.Tensor, node_id: int) -> Node:
        return Node(node_features, adjacency_matrix, node_id)
