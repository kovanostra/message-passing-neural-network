from unittest import TestCase

import torch as to
from torch import nn

from src.domain.graph import Graph
from src.domain.graph_encoder import GraphEncoder
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, \
    BASE_U_MATRIX, BASE_W_MATRIX, MULTIPLICATION_FACTOR, BASE_B_VECTOR


class TestGraphEncoder(TestCase):

    def setUp(self) -> None:
        graph = Graph(BASE_GRAPH,
                      BASE_GRAPH_NODE_FEATURES)
        self.graph_encoder = GraphEncoder(graph, initialize_tensors=False)
        self.graph_encoder.w_gru_update_gate_features = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX, requires_grad=True).float()
        self.graph_encoder.w_gru_forget_gate_features = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX, requires_grad=True).float()
        self.graph_encoder.w_gru_current_memory_message_features = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX, requires_grad=True).float()
        self.graph_encoder.u_gru_update_gate = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX, requires_grad=True).float()
        self.graph_encoder.u_gru_forget_gate = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX, requires_grad=True).float()
        self.graph_encoder.u_gru_current_memory_message = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX, requires_grad=True).float()
        self.graph_encoder.b_gru_update_gate = nn.Parameter(MULTIPLICATION_FACTOR * BASE_B_VECTOR, requires_grad=True).float()
        self.graph_encoder.b_gru_forget_gate = nn.Parameter(MULTIPLICATION_FACTOR * BASE_B_VECTOR, requires_grad=True).float()
        self.graph_encoder.b_gru_current_memory_message = nn.Parameter(MULTIPLICATION_FACTOR * BASE_B_VECTOR, requires_grad=True).float()
        self.graph_encoder.u_graph_node_features = nn.Parameter(0.1 * BASE_U_MATRIX, requires_grad=True).float()
        self.graph_encoder.u_graph_neighbor_messages = nn.Parameter(0.1 * BASE_U_MATRIX, requires_grad=True).float()

    def test_encode_graph_returns_the_expected_encoding_for_a_node_after_one_time_step(self):
        # Give
        self.graph_encoder.time_steps = 1
        node = 0
        node_encoding_expected = to.tensor([0.3909883, 0.3909883])
        graph = Graph(BASE_GRAPH,
                      BASE_GRAPH_NODE_FEATURES)

        # When
        node_encoding = self.graph_encoder.encode(graph)[node]

        # Then
        self.assertTrue(to.allclose(node_encoding_expected, node_encoding))

    def test_encode_graph_returns_the_expected_shape(self):
        # Given
        self.graph_encoder.time_steps = 1
        encoded_graph_shape_expected = BASE_GRAPH_NODE_FEATURES.shape
        graph = Graph(BASE_GRAPH,
                      BASE_GRAPH_NODE_FEATURES)

        # When
        encoded_graph_shape = self.graph_encoder.encode(graph).shape

        # Then
        self.assertEqual(encoded_graph_shape_expected, encoded_graph_shape)
