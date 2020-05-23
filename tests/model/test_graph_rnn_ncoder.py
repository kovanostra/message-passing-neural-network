from unittest import TestCase

import torch as to
from torch import nn
import numpy as np

from message_passing_nn.model.graph_rnn_encoder import GraphRNNEncoder
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, \
    MULTIPLICATION_FACTOR


class TestGraphRNNEncoder(TestCase):
    def setUp(self) -> None:
        self.number_of_nodes = BASE_GRAPH.size()[0]
        self.number_of_node_features = BASE_GRAPH_NODE_FEATURES.size()[1]
        self.fully_connected_layer_input_size = self.number_of_nodes * self.number_of_node_features
        self.fully_connected_layer_output_size = self.number_of_nodes ** 2
        self.device = "cpu"
        self.graph_encoder = GraphRNNEncoder(time_steps=2,
                                             number_of_nodes=self.number_of_nodes,
                                             number_of_node_features=self.number_of_node_features,
                                             fully_connected_layer_input_size=self.fully_connected_layer_input_size,
                                             fully_connected_layer_output_size=self.fully_connected_layer_output_size,
                                             device=self.device)
        self.graph_encoder.w_graph_node_features = nn.Parameter(
            MULTIPLICATION_FACTOR * (to.ones((self.number_of_node_features, self.number_of_node_features))),
            requires_grad=False)
        self.graph_encoder.w_graph_neighbor_messages = nn.Parameter(
            MULTIPLICATION_FACTOR * to.ones((self.number_of_node_features, self.number_of_node_features)),
            requires_grad=False)
        self.graph_encoder.u_graph_node_features = nn.Parameter(
            MULTIPLICATION_FACTOR * to.ones((self.number_of_nodes, self.number_of_nodes)),
            requires_grad=False)
        self.graph_encoder.u_graph_neighbor_messages = nn.Parameter(
            MULTIPLICATION_FACTOR * to.ones((self.number_of_node_features, self.number_of_node_features)),
            requires_grad=False)
        self.graph_encoder.linear.weight = to.nn.Parameter(
            MULTIPLICATION_FACTOR * to.ones(self.fully_connected_layer_output_size,
                                            self.fully_connected_layer_input_size),
            requires_grad=False).float()
        self.graph_encoder.linear.bias = to.nn.Parameter(
            MULTIPLICATION_FACTOR * to.tensor([i for i in range(self.fully_connected_layer_output_size)]),
            requires_grad=False).float()

    def test_forward_for_batch_size_one_and_two_steps(self):
        # Given
        batch_size = 1
        number_of_nodes = 4
        number_of_node_features = 2
        node_features = to.tensor([[[1.0, 2.0],
                                    [1.0, 1.0],
                                    [2.0, 0.5],
                                    [0.5, 0.5]]])
        adjacency_matrix = to.tensor([[[0.0, 1.0, 1.0, 0.0],
                                       [1.0, 0.0, 1.0, 0.0],
                                       [1.0, 1.0, 0.0, 1.0],
                                       [0.0, 0.0, 1.0, 0.0]]])

        # Calculations
        # -> Pre-loop
        batch = 0
        all_neighbors = [to.tensor([1, 2]), to.tensor([0, 2]), to.tensor([0, 1, 3]), to.tensor([2])]
        neighbors_slice = [[[2], [1]], [[2], [0]], [[1, 3], [0, 3], [0, 1]]]
        messages_init = to.zeros((self.number_of_nodes, self.number_of_nodes, self.number_of_node_features),
                                 device=self.device)

        # -> Step 0
        # Initialization
        messages_step_0_part_1 = to.zeros((self.number_of_nodes, self.number_of_nodes, self.number_of_node_features),
                                          device=self.device)
        messages_step_0_part_2 = to.zeros((self.number_of_nodes, self.number_of_nodes, self.number_of_node_features),
                                          device=self.device)

        # -> Step 0
        # Calculations
        index_pairs_expected = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 3]]
        index_pairs = []
        for node_id in range(number_of_nodes):
            messages_from_neighbors_step_0 = to.zeros((number_of_nodes, number_of_node_features), device=self.device)
            if batch + node_id < len(all_neighbors):
                for index in range(len(all_neighbors[batch + node_id])):
                    end_node_id = all_neighbors[batch + node_id][index].item()
                    messages_step_0_part_1[node_id, end_node_id] = self.graph_encoder.w_graph_node_features.matmul(
                        node_features[batch][node_id])
                    if batch + node_id < len(neighbors_slice):
                        index_pairs.append([node_id, end_node_id])
                        for node_to_sum in neighbors_slice[batch + node_id][index]:
                            messages_from_neighbors_step_0[
                                node_to_sum] = self.graph_encoder.w_graph_neighbor_messages.matmul(
                                messages_init[node_to_sum, node_id])
                        messages_step_0_part_2[node_id, end_node_id] = to.sum(messages_from_neighbors_step_0, dim=0)
        self.assertTrue(to.allclose(messages_step_0_part_1[0, 1], to.tensor([0.30, 0.30])))
        self.assertTrue(to.allclose(messages_step_0_part_1[1, 2], to.tensor([0.20, 0.20])))
        self.assertTrue(to.allclose(messages_step_0_part_1[2, 3], to.tensor([0.25, 0.25])))
        self.assertTrue(to.allclose(messages_step_0_part_1[3, 2], to.tensor([0.10, 0.10])))
        self.assertTrue(np.array_equal(index_pairs_expected, np.array(index_pairs)))
        self.assertTrue(to.allclose(messages_step_0_part_2, to.zeros(
            (self.number_of_nodes, self.number_of_nodes, self.number_of_node_features))))

        # -> Step 1
        # Messages
        messages_step_0 = to.relu(to.add(messages_step_0_part_1, messages_step_0_part_2))
        self.assertTrue(messages_step_0.size() == messages_init.size())
        print("Passed first step assertions!")

        # -> Step 1
        # Initialization
        messages_step_1_part_1 = to.zeros((self.number_of_nodes, self.number_of_nodes, self.number_of_node_features),
                                          device=self.device)
        messages_step_1_part_2 = to.zeros((self.number_of_nodes, self.number_of_nodes, self.number_of_node_features),
                                          device=self.device)

        # -> Step 1
        # Calculations
        index_pairs_expected = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 3]]
        index_pairs = []
        for node_id in range(number_of_nodes):
            messages_from_neighbors_step_1 = to.zeros((number_of_nodes, number_of_node_features), device=self.device)
            if batch + node_id < len(all_neighbors):
                for index in range(len(all_neighbors[batch + node_id])):
                    end_node_id = all_neighbors[batch + node_id][index].item()
                    messages_step_1_part_1[node_id, end_node_id] = self.graph_encoder.w_graph_node_features.matmul(
                        node_features[batch][node_id])
                    if batch + node_id < len(neighbors_slice):
                        index_pairs.append([node_id, end_node_id])
                        for node_to_sum in neighbors_slice[batch + node_id][index]:
                            messages_from_neighbors_step_1[
                                node_to_sum] = self.graph_encoder.w_graph_neighbor_messages.matmul(
                                messages_step_0[node_to_sum, node_id])
                        messages_step_1_part_2[node_id, end_node_id] = to.sum(messages_from_neighbors_step_1, dim=0)
        self.assertTrue(to.allclose(messages_step_0_part_1[0, 1], to.tensor([0.30, 0.30])))
        self.assertTrue(to.allclose(messages_step_0_part_1[1, 2], to.tensor([0.20, 0.20])))
        self.assertTrue(to.allclose(messages_step_0_part_1[2, 3], to.tensor([0.25, 0.25])))
        self.assertTrue(to.allclose(messages_step_0_part_1[3, 2], to.tensor([0.10, 0.10])))
        for node_id in range(number_of_nodes):
            for end_node_id in range(number_of_nodes):
                if [node_id, end_node_id] in index_pairs_expected:
                    self.assertTrue(messages_step_1_part_2[node_id, end_node_id][0].item() > 0.0)
                    self.assertTrue(messages_step_1_part_2[node_id, end_node_id][1].item() > 0.0)
                else:
                    self.assertTrue(to.allclose(messages_step_1_part_2[node_id, end_node_id], to.tensor([0.0, 0.0])))
        self.assertTrue(np.array_equal(index_pairs_expected, index_pairs))

        # -> Step 1
        # Messages
        messages_step_1 = to.relu(to.add(messages_step_1_part_1, messages_step_1_part_2))
        self.assertTrue(messages_step_1.size() == messages_init.size())
        print("Passed second step assertions!")

        # -> Sum messages
        messages_summed = to.zeros(number_of_nodes, number_of_node_features, device=self.device)
        index_pairs_expected = np.array([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 3], [3, 2]])
        index_pairs = []
        for node_id in range(number_of_nodes):
            if batch + node_id <= len(all_neighbors):
                messages_per_node = to.zeros(number_of_nodes, number_of_node_features, device=self.device)
                for index in range(len(all_neighbors[batch + node_id])):
                    end_node_id = all_neighbors[batch + node_id][index].item()
                    index_pairs.append([node_id, end_node_id])
                    messages_per_node[end_node_id] = self.graph_encoder.u_graph_neighbor_messages.matmul(
                        messages_step_1[end_node_id, node_id])
                messages_summed[node_id] = to.sum(messages_per_node, dim=0)
        self.assertTrue(messages_summed.size() == to.empty(number_of_nodes, number_of_node_features).size())
        self.assertTrue(np.array_equal(index_pairs_expected, np.array(index_pairs)))
        print("Passed sum messages assertions!")

        # -> Get encodings
        encodings = to.relu(to.add(self.graph_encoder.u_graph_node_features.matmul(node_features[batch]),
                                   messages_summed))
        self.assertTrue(encodings.size() == to.empty(number_of_nodes, number_of_node_features).size())
        print("Passed encodings assertions!")

        # -> Pass through fully connected layer
        weight = MULTIPLICATION_FACTOR * to.ones(self.fully_connected_layer_output_size,
                                                 self.fully_connected_layer_input_size)
        bias = MULTIPLICATION_FACTOR * to.tensor([i for i in range(self.fully_connected_layer_output_size)])
        outputs_expected = to.sigmoid(to.add(weight.matmul(encodings.view(batch_size, -1, 1)).squeeze(), bias))
        self.assertTrue(len(outputs_expected) == self.fully_connected_layer_output_size)
        print("Passed outputs assertions!")

        # When
        outputs = self.graph_encoder.forward(node_features, adjacency_matrix, batch_size)

        # Then
        print(outputs_expected)
        print(outputs)
        self.assertTrue(np.allclose(outputs_expected.numpy(), outputs.numpy(), atol=1e-02))

    def test_encode_graph_returns_the_expected_encoding_for_a_node_after_one_time_step(self):
        # Give
        node = 0
        node_encoding_expected = to.tensor([[0.5880, 0.5380]])

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
