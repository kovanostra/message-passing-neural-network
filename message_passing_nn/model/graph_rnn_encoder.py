import rnn_encoder_forward as rnn_cpp
import torch as to
import torch.nn as nn


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
        #TODO: Initialize linear weight and bias, as parameters and pass them inside forward to apply bmm()
        return rnn_cpp.forward(
            self.time_steps,
            self.number_of_nodes,
            self.number_of_node_features,
            batch_size,
            self.fully_connected_layer_output_size,
            self.w_graph_node_features,
            self.w_graph_neighbor_messages,
            self.u_graph_neighbor_messages,
            self.u_graph_node_features,
            node_features,
            adjacency_matrix,
            self.linear)
