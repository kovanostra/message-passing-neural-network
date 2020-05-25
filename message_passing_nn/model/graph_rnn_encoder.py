import torch as to
import torch.nn as nn

from message_passing_nn.data.data_preprocessor import DataPreprocessor
import rnn_encoder_forward as rnn_cpp


class GraphRNNEncoder(nn.Module):
    def __init__(self,
                 time_steps: int,
                 number_of_nodes: int,
                 number_of_node_features: int,
                 fully_connected_layer_input_size: int,
                 fully_connected_layer_output_size: int,
                 device: str) -> None:
        super(GraphRNNEncoder, self).__init__()

        self.time_steps = time_steps
        self.number_of_nodes = number_of_nodes
        self.number_of_node_features = number_of_node_features
        self.fully_connected_layer_input_size = fully_connected_layer_input_size
        self.fully_connected_layer_output_size = fully_connected_layer_output_size
        self.device = device

        self.w_graph_node_features = nn.Parameter(
            nn.init.kaiming_normal_(to.zeros([number_of_node_features, number_of_node_features], device=self.device)),
            requires_grad=True)
        self.w_graph_neighbor_messages = nn.Parameter(
            nn.init.kaiming_normal_(to.zeros([number_of_node_features, number_of_node_features], device=self.device)),
            requires_grad=True)
        self.u_graph_node_features = nn.Parameter(
            nn.init.kaiming_normal_(to.zeros([number_of_nodes, number_of_nodes], device=self.device)),
            requires_grad=True)
        self.u_graph_neighbor_messages = nn.Parameter(
            nn.init.kaiming_normal_(to.zeros([number_of_node_features, number_of_node_features], device=self.device)),
            requires_grad=True)
        self.linear = to.nn.Linear(self.fully_connected_layer_input_size, self.fully_connected_layer_output_size)
        self.sigmoid = to.nn.Sigmoid()

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
        messages = to.zeros((self.number_of_nodes, self.number_of_nodes, self.number_of_node_features),
                            device=self.device)
        messages = rnn_cpp.compose_messages(
            self.time_steps,
            self.number_of_nodes,
            self.number_of_node_features,
            self.w_graph_node_features,
            self.w_graph_neighbor_messages,
            node_features,
            adjacency_matrix,
            messages)
        node_encoding_messages = to.zeros(self.number_of_nodes, self.number_of_node_features, device=self.device)
        return rnn_cpp.encode_messages(
            self.number_of_nodes,
            node_encoding_messages,
            self.u_graph_neighbor_messages,
            self.u_graph_node_features,
            node_features,
            adjacency_matrix,
            messages)
