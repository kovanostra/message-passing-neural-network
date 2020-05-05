from unittest import TestCase

import torch as to
from torch import nn

from message_passing_nn.model.graph_gru_encoder import GraphGRUEncoder
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, \
    BASE_U_MATRIX, BASE_W_MATRIX, MULTIPLICATION_FACTOR, BASE_B_VECTOR


class TestGraphGRUEncoder(TestCase):

    def setUp(self) -> None:
        number_of_nodes = BASE_GRAPH.size()[0]
        number_of_node_features = BASE_GRAPH_NODE_FEATURES.size()[1]
        device = "cpu"
        self.graph_encoder = GraphGRUEncoder.of(time_steps=1,
                                                number_of_nodes=number_of_nodes,
                                                number_of_node_features=number_of_node_features,
                                                fully_connected_layer_input_size=number_of_nodes * number_of_node_features,
                                                fully_connected_layer_output_size=number_of_nodes ** 2,
                                                device=device)
        self.graph_encoder.w_gru_update_gate_features = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX,
                                                                     requires_grad=False).float()
        self.graph_encoder.w_gru_forget_gate_features = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX,
                                                                     requires_grad=False).float()
        self.graph_encoder.w_gru_current_memory_message_features = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX,
                                                                                requires_grad=False).float()
        self.graph_encoder.u_gru_update_gate = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX,
                                                            requires_grad=False).float()
        self.graph_encoder.u_gru_forget_gate = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX,
                                                            requires_grad=False).float()
        self.graph_encoder.u_gru_current_memory_message = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX,
                                                                       requires_grad=False).float()
        self.graph_encoder.b_gru_update_gate = nn.Parameter(MULTIPLICATION_FACTOR * BASE_B_VECTOR,
                                                            requires_grad=False).float()
        self.graph_encoder.b_gru_forget_gate = nn.Parameter(MULTIPLICATION_FACTOR * BASE_B_VECTOR,
                                                            requires_grad=False).float()
        self.graph_encoder.b_gru_current_memory_message = nn.Parameter(MULTIPLICATION_FACTOR * BASE_B_VECTOR,
                                                                       requires_grad=False).float()
        self.graph_encoder.u_graph_node_features = nn.Parameter(MULTIPLICATION_FACTOR * BASE_U_MATRIX,
                                                                requires_grad=False).float()
        self.graph_encoder.u_graph_neighbor_messages = nn.Parameter(MULTIPLICATION_FACTOR * BASE_U_MATRIX,
                                                                    requires_grad=False).float()
        self.graph_encoder.linear.weight = to.nn.Parameter(to.ones(number_of_nodes ** 2,
                                                                   number_of_nodes * number_of_node_features),
                                                           requires_grad=False).float()
        self.graph_encoder.linear.bias = to.nn.Parameter(2 * to.ones(number_of_nodes ** 2), requires_grad=False).float()

    def test_forward(self):
        # Given
        batch_size = 1
        features = BASE_GRAPH_NODE_FEATURES.view(1, BASE_GRAPH.size()[0], BASE_GRAPH_NODE_FEATURES.size()[1])
        adjacency_matrix = BASE_GRAPH.view(1, BASE_GRAPH.size()[0], BASE_GRAPH.size()[1],)
        outputs_expected = to.tensor([[0.98733, 0.98733, 0.98733, 0.98733, 0.98733, 0.98733, 0.98733, 0.98733, 0.98733,
                                       0.98733, 0.98733, 0.98733, 0.98733, 0.98733, 0.98733, 0.98733]])

        # When
        outputs = self.graph_encoder.forward(features, adjacency_matrix, batch_size)

        # Then
        self.assertTrue(to.allclose(outputs_expected, outputs))

    def test_encode_graph_returns_the_expected_encoding_for_a_node_after_one_time_step(self):
        # Give
        node = 0
        node_encoding_expected = to.tensor([[0.3909883, 0.3909883]])

        # When
        node_encoding = self.graph_encoder.encode(BASE_GRAPH_NODE_FEATURES, BASE_GRAPH)[node]

        # Then
        self.assertTrue(to.allclose(node_encoding_expected, node_encoding))

    def test_encode_graph_returns_the_expected_shape(self):
        # Given
        encoded_graph_shape_expected = list(BASE_GRAPH_NODE_FEATURES.shape)

        # When
        encoded_graph_shape = self.graph_encoder.encode(BASE_GRAPH_NODE_FEATURES, BASE_GRAPH).shape

        # Then
        self.assertEqual(encoded_graph_shape_expected, list(encoded_graph_shape))
