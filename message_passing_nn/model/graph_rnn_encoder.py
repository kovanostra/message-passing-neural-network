from typing import List, Any, Tuple

import graph_rnn_encoder_cpp as rnn_encoder_cpp
import math
import torch as to
import torch.nn as nn
from torch.nn import init


class GraphRNNEncoderFunction(to.autograd.Function):
    @staticmethod
    def forward(ctx,
                time_steps: int,
                number_of_nodes: int,
                number_of_node_features: int,
                fully_connected_layer_output_size: int,
                batch_size: int,
                node_features: to.Tensor,
                adjacency_matrix: to.Tensor,
                w_graph_node_features: to.Tensor,
                w_graph_neighbor_messages: to.Tensor,
                u_graph_node_features: to.Tensor,
                u_graph_neighbor_messages: to.Tensor,
                linear_weight: to.Tensor,
                linear_bias: to.Tensor) -> to.Tensor:
        outputs, linear_outputs, encodings, messages = rnn_encoder_cpp.forward(
            time_steps,
            number_of_nodes,
            number_of_node_features,
            fully_connected_layer_output_size,
            batch_size,
            node_features,
            adjacency_matrix,
            w_graph_node_features,
            w_graph_neighbor_messages,
            u_graph_node_features,
            u_graph_neighbor_messages,
            linear_weight,
            linear_bias)
        variables = [outputs,
                     linear_outputs,
                     encodings.view(batch_size, number_of_nodes*number_of_node_features),
                     to.sum(to.relu(messages), dim=2).squeeze(),
                     messages,
                     node_features,
                     to.Tensor([batch_size]),
                     to.Tensor([number_of_nodes]),
                     to.Tensor([number_of_node_features]),
                     w_graph_node_features,
                     w_graph_neighbor_messages,
                     u_graph_node_features,
                     u_graph_neighbor_messages,
                     linear_weight,
                     linear_bias]
        ctx.save_for_backward(*variables)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs: to.Tensor) -> Tuple[None,
                                                        None,
                                                        None,
                                                        None,
                                                        None,
                                                        None,
                                                        None,
                                                        to.Tensor,
                                                        to.Tensor,
                                                        to.Tensor,
                                                        to.Tensor,
                                                        to.Tensor,
                                                        to.Tensor]:
        backward_outputs = rnn_encoder_cpp.backward(grad_outputs.contiguous(), *ctx.saved_tensors)
        d_w_graph_node_features, d_w_graph_neighbor_messages, d_u_graph_neighbor_messages, d_u_graph_node_features, d_linear_weight, d_linear_bias = backward_outputs
        return None, \
               None, \
               None, \
               None, \
               None, \
               None, \
               None, \
               d_w_graph_node_features, \
               d_w_graph_neighbor_messages, \
               d_u_graph_neighbor_messages, \
               d_u_graph_node_features, \
               d_linear_weight, \
               d_linear_bias


class GraphRNNEncoder(nn.Module):
    def __init__(self,
                 time_steps: int,
                 number_of_nodes: int,
                 number_of_node_features: int,
                 fully_connected_layer_input_size: int,
                 fully_connected_layer_output_size: int) -> None:
        super(GraphRNNEncoder, self).__init__()

        self.time_steps = time_steps
        self.number_of_nodes = number_of_nodes
        self.number_of_node_features = number_of_node_features
        self.fully_connected_layer_input_size = fully_connected_layer_input_size
        self.fully_connected_layer_output_size = fully_connected_layer_output_size

        self.w_graph_node_features = nn.Parameter(
            to.empty([number_of_node_features, number_of_node_features]),
            requires_grad=True)
        self.w_graph_neighbor_messages = nn.Parameter(
            to.empty([number_of_node_features, number_of_node_features]),
            requires_grad=True)
        self.u_graph_node_features = nn.Parameter(
            to.empty([number_of_nodes, number_of_nodes]),
            requires_grad=True)
        self.u_graph_neighbor_messages = nn.Parameter(
            to.empty([number_of_node_features, number_of_node_features]),
            requires_grad=True)
        self.linear_weight = nn.Parameter(
            to.empty([self.fully_connected_layer_output_size, self.fully_connected_layer_input_size]),
            requires_grad=True)
        self.linear_bias = nn.Parameter(
            to.empty(self.fully_connected_layer_output_size),
            requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.w_graph_node_features)
        nn.init.kaiming_normal_(self.w_graph_neighbor_messages)
        nn.init.kaiming_normal_(self.u_graph_node_features)
        nn.init.kaiming_normal_(self.u_graph_neighbor_messages)
        nn.init.kaiming_uniform_(self.linear_weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.linear_weight)
        nn.init.uniform_(self.linear_bias, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))

    def forward(self,
                node_features: to.Tensor,
                adjacency_matrix: to.Tensor,
                batch_size: int) -> to.Tensor:
        return GraphRNNEncoderFunction.apply(self.time_steps,
                                             self.number_of_nodes,
                                             self.number_of_node_features,
                                             self.fully_connected_layer_output_size,
                                             batch_size,
                                             node_features,
                                             adjacency_matrix,
                                             self.w_graph_node_features,
                                             self.w_graph_neighbor_messages,
                                             self.u_graph_node_features,
                                             self.u_graph_neighbor_messages,
                                             self.linear_weight,
                                             self.linear_bias)
