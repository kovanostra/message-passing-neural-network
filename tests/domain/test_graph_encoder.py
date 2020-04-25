from unittest import TestCase

import torch as to
from torch import nn
from torch.utils.data import DataLoader

from src.domain.graph_dataset import GraphDataset
from src.domain.graph_encoder import GraphEncoder
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, \
    BASE_U_MATRIX, BASE_W_MATRIX, MULTIPLICATION_FACTOR, BASE_B_VECTOR


class TestGraphEncoder(TestCase):

    def setUp(self) -> None:
        self.graph_encoder = GraphEncoder(time_steps=1,
                                          number_of_nodes=BASE_GRAPH.size()[0],
                                          number_of_node_features=BASE_GRAPH_NODE_FEATURES.size()[1])
        self.graph_encoder.w_gru_update_gate_features = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX,
                                                                     requires_grad=True).float()
        self.graph_encoder.w_gru_forget_gate_features = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX,
                                                                     requires_grad=True).float()
        self.graph_encoder.w_gru_current_memory_message_features = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX,
                                                                                requires_grad=True).float()
        self.graph_encoder.u_gru_update_gate = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX,
                                                            requires_grad=True).float()
        self.graph_encoder.u_gru_forget_gate = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX,
                                                            requires_grad=True).float()
        self.graph_encoder.u_gru_current_memory_message = nn.Parameter(MULTIPLICATION_FACTOR * BASE_W_MATRIX,
                                                                       requires_grad=True).float()
        self.graph_encoder.b_gru_update_gate = nn.Parameter(MULTIPLICATION_FACTOR * BASE_B_VECTOR,
                                                            requires_grad=True).float()
        self.graph_encoder.b_gru_forget_gate = nn.Parameter(MULTIPLICATION_FACTOR * BASE_B_VECTOR,
                                                            requires_grad=True).float()
        self.graph_encoder.b_gru_current_memory_message = nn.Parameter(MULTIPLICATION_FACTOR * BASE_B_VECTOR,
                                                                       requires_grad=True).float()
        self.graph_encoder.u_graph_node_features = nn.Parameter(0.1 * BASE_U_MATRIX, requires_grad=True).float()
        self.graph_encoder.u_graph_neighbor_messages = nn.Parameter(0.1 * BASE_U_MATRIX, requires_grad=True).float()

    def test_encode_graph_returns_the_expected_encoding_for_a_node_after_one_time_step(self):
        # Give
        node = 0
        batch_size = 1
        node_features = BASE_GRAPH_NODE_FEATURES
        adjacency_matrix = BASE_GRAPH
        raw_training_data = [(node_features, adjacency_matrix)]
        graph_dataset = GraphDataset(raw_training_data)
        training_data = DataLoader(graph_dataset, batch_size)
        node_encoding_expected = to.tensor([[0.3909883, 0.3909883]])

        # When
        for features, labels in training_data:
            encoded_graph = self.graph_encoder.forward(features, adjacency_matrix=labels, batch_size=batch_size)
        node_encoding = encoded_graph[:, node]

        # Then
        self.assertTrue(to.allclose(node_encoding_expected, node_encoding))

    def test_encode_graph_returns_the_expected_shape(self):
        # Given
        batch_size = 1
        node_features = BASE_GRAPH_NODE_FEATURES
        adjacency_matrix = BASE_GRAPH
        raw_training_data = [(node_features, adjacency_matrix)]
        graph_dataset = GraphDataset(raw_training_data)
        training_data = DataLoader(graph_dataset, batch_size)
        encoded_graph_shape_expected = [batch_size] + list(BASE_GRAPH_NODE_FEATURES.shape)

        # When
        for features, labels in training_data:
            encoded_graph = self.graph_encoder.forward(features, adjacency_matrix=labels, batch_size=batch_size)
        encoded_graph_shape = list(encoded_graph.shape)

        # Then
        self.assertEqual(encoded_graph_shape_expected, encoded_graph_shape)
