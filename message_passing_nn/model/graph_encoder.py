import torch as to
import torch.nn as nn

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.model.edge import Edge
from message_passing_nn.model.graph import Graph
from message_passing_nn.model.message_gru import MessageGRU
from message_passing_nn.model.node import Node


class GraphEncoder(nn.Module):
    def __init__(self,
                 time_steps: int,
                 number_of_nodes: int,
                 number_of_node_features: int,
                 fully_connected_layer_input_size: int,
                 fully_connected_layer_output_size: int,
                 device: str) -> None:
        super(GraphEncoder, self).__init__()
        self.time_steps = time_steps
        self.number_of_nodes = number_of_nodes
        self.number_of_node_features = number_of_node_features
        self.fully_connected_layer_input_size = fully_connected_layer_input_size
        self.fully_connected_layer_output_size = fully_connected_layer_output_size
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
        self.device = device
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

    def forward(self, node_features: to.Tensor, adjacency_matrix: to.Tensor, batch_size: int) -> to.Tensor:
        outputs = to.zeros(batch_size, self.fully_connected_layer_output_size).to(self.device)
        for batch in range(batch_size):
            outputs[batch] = self.sigmoid(
                self.linear(
                    DataPreprocessor.flatten(
                        self.encode(node_features[batch], adjacency_matrix[batch]))))
        return outputs

    def encode(self, node_features: to.Tensor, adjacency_matrix: to.Tensor) -> to.Tensor:
        messages = self._send_messages(node_features, adjacency_matrix)
        encodings = self._encode_nodes(node_features, messages)
        return encodings

    def initialize_tensors(self, graph: Graph) -> None:
        base_4d_tensor_shape = [graph.number_of_nodes,
                                graph.number_of_nodes,
                                graph.number_of_node_features,
                                graph.number_of_node_features]
        base_3d_tensor_shape = [graph.number_of_nodes,
                                graph.number_of_node_features,
                                graph.number_of_node_features]
        base_2d_tensor_shape = [graph.number_of_node_features]
        self.w_gru_update_gate_features = nn.Parameter(to.rand(base_4d_tensor_shape), requires_grad=True).to(self.device)
        self.w_gru_forget_gate_features = nn.Parameter(to.rand(base_4d_tensor_shape), requires_grad=True).to(self.device)
        self.w_gru_current_memory_message_features = nn.Parameter(to.rand(base_4d_tensor_shape), requires_grad=True).to(self.device)
        self.u_gru_update_gate = nn.Parameter(to.rand(base_4d_tensor_shape), requires_grad=True).to(self.device)
        self.u_gru_forget_gate = nn.Parameter(to.rand(base_4d_tensor_shape), requires_grad=True).to(self.device)
        self.u_gru_current_memory_message = nn.Parameter(to.rand(base_4d_tensor_shape), requires_grad=True).to(self.device)
        self.b_gru_update_gate = nn.Parameter(to.rand(base_2d_tensor_shape), requires_grad=True).to(self.device)
        self.b_gru_forget_gate = nn.Parameter(to.rand(base_2d_tensor_shape), requires_grad=True).to(self.device)
        self.b_gru_current_memory_message = nn.Parameter(to.rand(base_2d_tensor_shape), requires_grad=True).to(self.device)
        self.u_graph_node_features = nn.Parameter(to.rand(base_3d_tensor_shape), requires_grad=True).to(self.device)
        self.u_graph_neighbor_messages = nn.Parameter(to.rand(base_3d_tensor_shape), requires_grad=True).to(self.device)

    def _send_messages(self, node_features: to.Tensor, adjacency_matrix: to.Tensor) -> to.Tensor:
        messages = to.zeros((self.number_of_nodes, self.number_of_nodes, self.number_of_node_features)).to(self.device)
        for step in range(self.time_steps):
            messages = self._compose_messages_from_nodes_to_targets(node_features,
                                                                    adjacency_matrix,
                                                                    messages)
        return messages

    def _encode_nodes(self, node_features: to.Tensor, messages: to.Tensor) -> to.Tensor:
        encoded_node = to.zeros(self.number_of_nodes, self.number_of_node_features).to(self.device)
        for node_id in range(self.number_of_nodes):
            encoded_node[node_id] += self._apply_recurrent_layer_for_node(node_features,
                                                                          messages,
                                                                          node_id)
        return encoded_node

    def _apply_recurrent_layer_for_node(self, node_features: to.Tensor, messages: to.Tensor, node_id: int) -> to.Tensor:
        node_encoding_features = self.u_graph_node_features[node_id].matmul(node_features[node_id])
        node_encoding_messages = self.u_graph_neighbor_messages[node_id].matmul(to.sum(messages[node_id], dim=0))
        return to.relu(node_encoding_features + node_encoding_messages)

    def _compose_messages_from_nodes_to_targets(self,
                                                node_features: to.Tensor,
                                                adjacency_matrix: to.Tensor,
                                                messages: to.Tensor) -> to.Tensor:
        new_messages = to.zeros(messages.shape).to(self.device)
        for node_id in range(self.number_of_nodes):
            node = self._create_node(node_features, adjacency_matrix, node_id)
            for end_node_id in node.neighbors:
                end_node = self._create_node(node_features, adjacency_matrix, end_node_id)
                edge = self._create_edge(node, end_node)
                edge_slice = edge.get_edge_slice()
                message = self._get_message_inputs(messages, node, edge, node_features, adjacency_matrix)
                message.compose()
                new_messages[edge_slice] = message.value
        return new_messages

    def _get_message_inputs(self,
                            messages: to.Tensor,
                            node: Node,
                            edge: Edge,
                            node_features: to.Tensor,
                            adjacency_matrix: to.Tensor) -> MessageGRU:
        message = self._create_message(self.device)
        message.previous_messages = self._get_messages_from_all_node_neighbors_except_target_summed(messages,
                                                                                                    node,
                                                                                                    edge)
        message.update_gate = self._pass_through_update_gate(messages, node, edge, node_features)
        message.current_memory = self._get_current_memory_message(messages, node, edge, node_features, adjacency_matrix)
        return message

    def _get_messages_from_all_node_neighbors_except_target_summed(self,
                                                                   messages: to.Tensor,
                                                                   node: Node,
                                                                   edge: Edge) -> to.Tensor:
        messages_from_the_other_neighbors = to.zeros(node.features.shape[0]).to(self.device)
        if node.neighbors_count > 1:
            neighbors_slice = edge.get_start_node_neighbors_without_end_node()
            messages_from_the_other_neighbors = to.sum(messages[neighbors_slice], dim=0)
        return messages_from_the_other_neighbors

    def _pass_through_update_gate(self,
                                  messages: to.Tensor,
                                  node: Node,
                                  edge: Edge,
                                  node_features: to.Tensor) -> to.Tensor:
        message_from_a_neighbor_other_than_target = self._get_messages_from_all_node_neighbors_except_target_summed(
            messages,
            node,
            edge)
        edge_slice = edge.get_edge_slice()
        update_gate_output = to.sigmoid(
            to.add(
                to.add(self.w_gru_update_gate_features[edge_slice].matmul(node_features[node.node_id]),
                       self.u_gru_update_gate[edge_slice].matmul(message_from_a_neighbor_other_than_target)),
                self.b_gru_update_gate))
        return update_gate_output

    def _get_current_memory_message(self,
                                    messages: to.Tensor,
                                    node: Node,
                                    edge: Edge,
                                    node_features: to.Tensor,
                                    adjacency_matrix: to.Tensor) -> to.Tensor:
        messages_passed_through_reset_gate = self._keep_or_reset_messages(messages,
                                                                          node,
                                                                          edge,
                                                                          node_features,
                                                                          adjacency_matrix)
        edge_slice = edge.get_edge_slice()
        current_memory_message = to.add(
            to.add(self.w_gru_current_memory_message_features[edge_slice].matmul(node_features[node.node_id]),
                   self.u_gru_current_memory_message[edge_slice].matmul(messages_passed_through_reset_gate)),
            self.b_gru_current_memory_message)
        return to.tanh(current_memory_message)

    def _keep_or_reset_messages(self,
                                messages: to.Tensor,
                                node: Node,
                                edge: Edge,
                                node_features: to.Tensor,
                                adjacency_matrix: to.Tensor) -> to.Tensor:
        edge_slice = edge.get_edge_slice()
        neighbors_slice = edge.get_start_node_neighbors_without_end_node()[0]
        messages_from_the_other_neighbors = to.zeros(node.features.shape[0]).to(self.device)
        for reset_node_index in neighbors_slice:
            reset_node = self._create_node(node_features, adjacency_matrix, reset_node_index)
            reset_edge = self._create_edge(node, reset_node)
            reset_edge_slice = reset_edge.get_edge_slice()
            reset_gate_output = self._pass_through_reset_gate(messages, node, reset_edge, node_features)
            messages_from_the_other_neighbors += to.mul(reset_gate_output, messages[reset_edge_slice])
        return self.u_gru_current_memory_message[edge_slice].matmul(messages_from_the_other_neighbors)

    def _pass_through_reset_gate(self,
                                 messages: to.Tensor,
                                 node: Node,
                                 edge: Edge,
                                 node_features: to.Tensor) -> to.Tensor:
        edge_slice = edge.get_edge_slice()
        message_from_a_neighbor_other_than_target = messages[edge_slice]
        return to.sigmoid(
            to.add(
                to.add(self.w_gru_update_gate_features[edge_slice].matmul(node_features[node.node_id]),
                       self.u_gru_update_gate[edge_slice].matmul(message_from_a_neighbor_other_than_target)),
                self.b_gru_update_gate)).long()

    @staticmethod
    def _create_node(node_features: to.Tensor, adjacency_matrix: to.Tensor, node_id: to.Tensor) -> Node:
        return Node(node_features, adjacency_matrix, node_id)

    @staticmethod
    def _create_edge(start_node: Node, end_node: Node) -> Edge:
        return Edge(start_node, end_node)

    @staticmethod
    def _create_message(device: str) -> MessageGRU:
        return MessageGRU(device)
