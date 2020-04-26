from unittest import TestCase

from src.domain.data_preprocessor import DataPreprocessor
from src.domain.graph import Graph
from src.domain.graph_encoder import GraphEncoder
from src.domain.loss_function_selector import LossFunctionSelector
from src.domain.model_trainer import ModelTrainer
from src.domain.optimizer_selector import OptimizerSelector
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES


class TestModelTrainer(TestCase):
    def setUp(self) -> None:
        loss_function = LossFunctionSelector("MSE").loss_function
        optimizer = OptimizerSelector("SGD").optimizer
        self.model_trainer = ModelTrainer(GraphEncoder, loss_function, optimizer)

    def test_instantiate_model_and_optimizer(self):
        # Given
        initialization_graph = Graph(BASE_GRAPH, BASE_GRAPH_NODE_FEATURES)

        # When
        self.model_trainer.instantiate_model_and_optimizer(initialization_graph)

        # Then
        self.assertTrue(self.model_trainer.model.number_of_nodes == initialization_graph.number_of_nodes)
        self.assertTrue(self.model_trainer.model.number_of_node_features == initialization_graph.number_of_node_features)
        self.assertTrue(self.model_trainer.optimizer.param_groups)

    def test_do_train(self):
        # Given
        self.model_trainer.instantiate_model_and_optimizer(Graph(BASE_GRAPH, BASE_GRAPH_NODE_FEATURES))
        raw_dataset = [(BASE_GRAPH_NODE_FEATURES, BASE_GRAPH)]
        training_data, _, _ = DataPreprocessor.train_validation_test_split(raw_dataset, 1, 0.0, 0.0)

        # When
        training_loss = self.model_trainer.do_train(training_data=training_data, epoch=1)

        # Then
        self.assertTrue(training_loss > 0.0)

    def test_do_evaluate(self):
        # Given
        self.model_trainer.instantiate_model_and_optimizer(Graph(BASE_GRAPH, BASE_GRAPH_NODE_FEATURES))
        raw_dataset = [(BASE_GRAPH_NODE_FEATURES, BASE_GRAPH)]
        training_data, _, _ = DataPreprocessor.train_validation_test_split(raw_dataset, 1, 0.0, 0.0)

        # When
        validation_loss = self.model_trainer.do_evaluate(evaluation_data=training_data, epoch=1)

        # Then
        self.assertTrue(validation_loss > 0.0)
