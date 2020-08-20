from typing import Tuple

import rnn_encoder_cpp as rnn_encoder_cpp
try:
    import rnn_encoder_cuda_cpp as rnn_encoder_cuda_cpp
except:
    pass
import math
import torch as to
import torch.nn as nn
from torch.nn import init


class RNNEncoderFunction(to.autograd.Function):
    @staticmethod
    def forward(ctx,
                time_steps: int,
                number_of_nodes: int,
                number_of_node_features: int,
                fully_connected_layer_output_size: int,
                batch_size: int,
                device: str,
                node_features: to.Tensor,
                all_neighbors: to.Tensor,
                w_graph_node_features: to.Tensor,
                w_graph_neighbor_messages: to.Tensor,
                u_graph_node_features: to.Tensor,
                u_graph_neighbor_messages: to.Tensor,
                linear_weight: to.Tensor,
                linear_bias: to.Tensor) -> to.Tensor:
        if device == "cuda":
            cpp_extension = rnn_encoder_cuda_cpp
        else:
            cpp_extension = rnn_encoder_cpp
        outputs, linear_outputs, encodings, messages, messages_previous_step = cpp_extension.forward(
            to.tensor(time_steps, device=device),
            to.tensor(number_of_nodes, device=device),
            to.tensor(number_of_node_features, device=device),
            to.tensor(fully_connected_layer_output_size, device=device),
            to.tensor(batch_size, device=device),
            node_features,
            all_neighbors,
            w_graph_node_features,
            w_graph_neighbor_messages,
            u_graph_node_features,
            u_graph_neighbor_messages,
            linear_weight,
            linear_bias)
        variables = [outputs,
                     linear_outputs,
                     encodings.view(batch_size, number_of_nodes * number_of_node_features),
                     to.sum(to.relu(messages), dim=2).squeeze(),
                     to.sum(to.relu(messages_previous_step), dim=2).squeeze(),
                     messages,
                     node_features,
                     to.tensor([batch_size]),
                     to.tensor([number_of_nodes]),
                     to.tensor([number_of_node_features]),
                     to.sum(u_graph_neighbor_messages, dim=0),
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
                                                        None,
                                                        to.Tensor,
                                                        to.Tensor,
                                                        to.Tensor,
                                                        to.Tensor,
                                                        to.Tensor,
                                                        to.Tensor]:
        if grad_outputs.device == "cuda":
            cpp_extension = rnn_encoder_cuda_cpp
        else:
            cpp_extension = rnn_encoder_cpp
        backward_outputs = cpp_extension.backward(grad_outputs.contiguous(), *ctx.saved_tensors)
        d_w_graph_node_features, d_w_graph_neighbor_messages, d_u_graph_neighbor_messages, d_u_graph_node_features, d_linear_weight, d_linear_bias = backward_outputs
        return None, \
               None, \
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


class RNNEncoder(nn.Module):
    def __init__(self,
                 time_steps: int,
                 number_of_nodes: int,
                 number_of_node_features: int,
                 fully_connected_layer_input_size: int,
                 fully_connected_layer_output_size: int,
                 device: str = "cpu") -> None:
        super(RNNEncoder, self).__init__()

        self.time_steps = time_steps
        self.number_of_nodes = number_of_nodes
        self.number_of_node_features = number_of_node_features
        self.fully_connected_layer_input_size = fully_connected_layer_input_size
        self.fully_connected_layer_output_size = fully_connected_layer_output_size
        self.device = device

        self.w_graph_node_features = nn.Parameter(
            to.empty([number_of_nodes, number_of_nodes],
                     device=self.device),
            requires_grad=True)
        self.w_graph_neighbor_messages = nn.Parameter(
            to.empty([number_of_nodes, number_of_nodes],
                     device=self.device),
            requires_grad=True)
        self.u_graph_node_features = nn.Parameter(
            to.empty([number_of_nodes, number_of_nodes],
                     device=self.device),
            requires_grad=True)
        self.u_graph_neighbor_messages = nn.Parameter(
            to.empty([number_of_node_features, number_of_node_features],
                     device=self.device),
            requires_grad=True)
        self.linear_weight = nn.Parameter(
            to.empty([self.fully_connected_layer_output_size, self.fully_connected_layer_input_size],
                     device=self.device),
            requires_grad=True)
        self.linear_bias = nn.Parameter(
            to.empty(self.fully_connected_layer_output_size,
                     device=self.device),
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
                all_neighbors: to.Tensor,
                batch_size: int) -> to.Tensor:
        return RNNEncoderFunction.apply(self.time_steps,
                                        self.number_of_nodes,
                                        self.number_of_node_features,
                                        self.fully_connected_layer_output_size,
                                        batch_size,
                                        self.device,
                                        node_features,
                                        all_neighbors,
                                        self.w_graph_node_features,
                                        self.w_graph_neighbor_messages,
                                        self.u_graph_node_features,
                                        self.u_graph_neighbor_messages,
                                        self.linear_weight,
                                        self.linear_bias)

    def get_model_size(self) -> str:
        return str(int((self.w_graph_node_features.element_size() * self.w_graph_node_features.nelement() +
                        self.w_graph_neighbor_messages.element_size() * self.w_graph_neighbor_messages.nelement() +
                        self.u_graph_node_features.element_size() * self.u_graph_node_features.nelement() +
                        self.u_graph_neighbor_messages.element_size() * self.u_graph_neighbor_messages.nelement() +
                        self.linear_weight.element_size() * self.linear_weight.nelement() +
                        self.linear_bias.element_size() * self.linear_bias.nelement()) * 0.000001))
