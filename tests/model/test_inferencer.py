from unittest import TestCase

import torch as to
from message_passing_nn.data.graph_dataset import GraphDataset

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.model import Inferencer
from message_passing_nn.utils.model_selector import ModelSelector
from tests.fixtures.matrices_and_vectors import BASE_GRAPH_NODE_FEATURES, BASE_GRAPH


class TestInferencer(TestCase):
    def test_do_inference(self):
        # Given
        data_preprocessor = DataPreprocessor()
        device = "cpu"
        inferencer = Inferencer(data_preprocessor, device)
        data_dimensions = (BASE_GRAPH_NODE_FEATURES.size(), BASE_GRAPH.size(), BASE_GRAPH.view(-1).size())
        model = ModelSelector.load_model("RNN")
        model = model(time_steps=1,
                      number_of_nodes=data_dimensions[1][0],
                      number_of_node_features=data_dimensions[0][1],
                      fully_connected_layer_input_size=data_dimensions[1][0] * data_dimensions[0][1],
                      fully_connected_layer_output_size=data_dimensions[2][0])
        all_neighbors = to.tensor([[1, 2, -1, -1],
                                   [0, 2, -1, -1],
                                   [0, 1, 3, -1],
                                   [2, -1, -1, -1]])
        dataset = GraphDataset("")
        dataset.enable_test_mode()
        dataset.dataset = [(BASE_GRAPH_NODE_FEATURES, all_neighbors, BASE_GRAPH.view(-1))]
        inference_data, _, _ = DataPreprocessor().train_validation_test_split(dataset, 1, 0.0, 0.0)
        output_label_pairs_expected = [BASE_GRAPH.view(-1), BASE_GRAPH.view(-1)]

        # When
        output_label_pairs = inferencer.do_inference(model, inference_data)

        # Then
        self.assertEqual(output_label_pairs[0][0].squeeze().size(), output_label_pairs_expected[0].size())
        self.assertEqual(output_label_pairs[0][1].squeeze().size(), output_label_pairs_expected[1].size())
