from unittest import TestCase

import torch as to
from torch import nn

from message_passing_nn.model.graph_rnn_encoder import GraphRNNEncoder
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, BASE_W_MATRIX, BASE_U_MATRIX, \
    MULTIPLICATION_FACTOR


class TestGraphRNNEncoder(TestCase):
    def setUp(self) -> None:
        self.number_of_nodes = BASE_GRAPH.size()[0]
        self.number_of_node_features = BASE_GRAPH_NODE_FEATURES.size()[1]
        self.device = "cpu"
        self.graph_encoder = GraphRNNEncoder.of(time_steps=1,
                                                number_of_nodes=self.number_of_nodes,
                                                number_of_node_features=self.number_of_node_features,
                                                fully_connected_layer_input_size=self.number_of_nodes * self.number_of_node_features,
                                                fully_connected_layer_output_size=self.number_of_nodes ** 2,
                                                device=self.device)
        self.graph_encoder.w_graph_node_features = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX,
                                                                requires_grad=False).float()
        self.graph_encoder.w_graph_neighbor_messages = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX,
                                                                    requires_grad=False).float()
        self.graph_encoder.u_graph_node_features = nn.Parameter(MULTIPLICATION_FACTOR * BASE_U_MATRIX,
                                                                requires_grad=False).float()
        self.graph_encoder.u_graph_neighbor_messages = nn.Parameter(MULTIPLICATION_FACTOR * BASE_U_MATRIX,
                                                                    requires_grad=False).float()
        self.graph_encoder.linear.weight = to.nn.Parameter(to.ones(self.number_of_nodes ** 2,
                                                                   self.number_of_nodes * self.number_of_node_features),
                                                           requires_grad=False).float()
        self.graph_encoder.linear.bias = to.nn.Parameter(2 * to.ones(self.number_of_nodes ** 2),
                                                         requires_grad=False).float()

    def test_forward(self):
        # Given
        batch_size = 1
        features = BASE_GRAPH_NODE_FEATURES.view(1, BASE_GRAPH.size()[0], BASE_GRAPH_NODE_FEATURES.size()[1])
        adjacency_matrix = BASE_GRAPH.view(1, BASE_GRAPH.size()[0], BASE_GRAPH.size()[1], )
        outputs_expected = to.tensor([[0.98943, 0.98943, 0.98943, 0.98943, 0.98943, 0.98943, 0.98943, 0.98943, 0.98943,
                                       0.98943, 0.98943, 0.98943, 0.98943, 0.98943, 0.98943, 0.98943]])

        # When
        outputs = self.graph_encoder.forward(features, adjacency_matrix, batch_size)

        # Then
        self.assertTrue(to.allclose(outputs_expected, outputs))

    def test_encode_graph_returns_the_expected_encoding_for_a_node_after_one_time_step(self):
        # Give
        node = 0
        node_encoding_expected = to.tensor([[0.42, 0.42]])

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

    def test_get_the_expected_messages_from_a_node_after_one_time_step(self):
        messages_initial = to.zeros((self.number_of_nodes,
                                     self.number_of_nodes,
                                     self.number_of_node_features),
                                    device=self.device)
        node_expected = 0
        messages_from_node_expected = to.tensor([[0.0, 0.0],
                                                 [0.3, 0.3],
                                                 [0.3, 0.3],
                                                 [0.0, 0.0]])
        # When
        messages_from_node = self.graph_encoder._compose_messages(BASE_GRAPH_NODE_FEATURES,
                                                                  BASE_GRAPH,
                                                                  messages_initial)[node_expected]

        # Then
        self.assertTrue(to.allclose(messages_from_node_expected, messages_from_node))
