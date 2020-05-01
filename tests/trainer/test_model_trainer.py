from unittest import TestCase

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.model.graph import Graph
from message_passing_nn.model.graph_encoder import GraphEncoder
from message_passing_nn.trainer.model_trainer import ModelTrainer
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES


class TestModelTrainer(TestCase):
    def setUp(self) -> None:
        time_steps = 1
        loss_function = "MSE"
        optimizer = "SGD"
        device = "cpu"
        self.configuration_dictionary = {"loss_function": loss_function,
                                         "optimizer": optimizer,
                                         "time_steps": time_steps}
        self.model_trainer = ModelTrainer(GraphEncoder, device)

    def test_instantiate_attributes(self):
        # Given
        initialization_graph = Graph(BASE_GRAPH, BASE_GRAPH_NODE_FEATURES)

        # When
        self.model_trainer.instantiate_attributes(initialization_graph, self.configuration_dictionary)

        # Then
        self.assertTrue(self.model_trainer.model.number_of_nodes == initialization_graph.number_of_nodes)
        self.assertTrue(
            self.model_trainer.model.number_of_node_features == initialization_graph.number_of_node_features)
        self.assertTrue(self.model_trainer.optimizer.param_groups)

    def test_do_train(self):
        # Given
        self.model_trainer.instantiate_attributes(Graph(BASE_GRAPH, BASE_GRAPH_NODE_FEATURES),
                                                  self.configuration_dictionary)
        raw_dataset = [(BASE_GRAPH_NODE_FEATURES, BASE_GRAPH)]
        training_data, _, _ = DataPreprocessor.train_validation_test_split(raw_dataset, 1, 0.0, 0.0)

        # When
        training_loss = self.model_trainer.do_train(training_data=training_data, epoch=1)

        # Then
        self.assertTrue(training_loss > 0.0)

    def test_do_evaluate(self):
        # Given
        self.model_trainer.instantiate_attributes(Graph(BASE_GRAPH, BASE_GRAPH_NODE_FEATURES),
                                                  self.configuration_dictionary)
        raw_dataset = [(BASE_GRAPH_NODE_FEATURES, BASE_GRAPH)]
        training_data, _, _ = DataPreprocessor.train_validation_test_split(raw_dataset, 1, 0.0, 0.0)

        # When
        validation_loss = self.model_trainer.do_evaluate(evaluation_data=training_data, epoch=1)

        # Then
        self.assertTrue(validation_loss > 0.0)
