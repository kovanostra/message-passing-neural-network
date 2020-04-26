import logging
from typing import Any

import torch
from torch import nn

from src.domain.data_preprocessor import DataPreprocessor
from src.domain.graph import Graph
from src.domain.graph_encoder import GraphEncoder
from src.repository.interface.repository import Repository


class Training:
    def __init__(self, training_data_repository: Repository, epochs: int, loss_function: Any, optimizer: Any) -> None:
        self.repository = training_data_repository
        self.epochs = epochs
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.running_loss = 0.0

    def start(self, batch_size: int, validation_split: float, test_split: float):
        training_data, validation_data, test_data, initialization_graph = self._prepare_dataset(batch_size,
                                                                                                validation_split,
                                                                                                test_split)
        model = self._instantiate_model(initialization_graph)
        self._instantiate_the_optimizer(model)
        self.get_logger().info('Started Training')
        for epoch in range(self.epochs):
            self.running_loss = 0.0
            for features, labels in training_data:
                self._do_train(epoch, model, features, labels)
            if epoch % 10 == 0:
                validation_loss = self._do_evaluate(model, validation_data)
                self.get_logger().info("The validation loss at iteration " + str(epoch) + " is: " +
                                       str(round(validation_loss, 3)))
        test_loss = self._do_evaluate(model, test_data)
        self.get_logger().info("The test loss is: " + str(round(test_loss, 3)))
        self.get_logger().info('Finished Training')

    def _do_train(self,
                  epoch: int,
                  model: nn.Module,
                  features: Any,
                  labels: Any) -> Any:
        current_batch_size = self._get_current_batch_size(features)
        self.optimizer.zero_grad()
        outputs = model.forward(features, adjacency_matrix=labels, batch_size=current_batch_size)
        loss = self.loss_function(outputs, labels)
        self.running_loss = self._do_backpropagate(epoch, loss, self.running_loss)
        return self.running_loss

    def _do_backpropagate(self, epoch: int, loss: Any, running_loss: float) -> float:
        loss.backward()
        self.optimizer.step()
        running_loss += loss.item()
        self.get_logger().info('[%d] loss: %.3f' % (epoch + 1, running_loss))
        return running_loss

    def _do_evaluate(self,
                     model: nn.Module,
                     validation_data: Any) -> float:
        validation_loss = 0.0
        with torch.no_grad():
            for features_validation, labels_validation in validation_data:
                model.eval()
                current_batch_size = self._get_current_batch_size(features_validation)
                outputs = model.forward(features_validation, labels_validation, current_batch_size)
                loss = self.loss_function(outputs, labels_validation)
                validation_loss += float(loss)
            validation_loss /= len(validation_data)
        return validation_loss

    def _prepare_dataset(self, batch_size: int, validation_split: float, test_split: float) -> Any:
        raw_dataset = self.repository.get_all_features_and_labels_from_separate_files()
        training_data, validation_data, test_data = DataPreprocessor.train_validation_test_split(raw_dataset,
                                                                                                 batch_size,
                                                                                                 validation_split,
                                                                                                 test_split)
        initialization_graph = DataPreprocessor.extract_initialization_graph(raw_dataset)
        return training_data, validation_data, test_data, initialization_graph

    @staticmethod
    def _instantiate_model(initialization_graph: Graph) -> GraphEncoder:
        number_of_nodes = initialization_graph.number_of_nodes
        number_of_node_features = initialization_graph.number_of_node_features
        model = GraphEncoder(time_steps=2,
                             number_of_nodes=number_of_nodes,
                             number_of_node_features=number_of_node_features,
                             fully_connected_layer_input_size=number_of_nodes * number_of_node_features,
                             fully_connected_layer_output_size=number_of_nodes ** 2)
        model.initialize_tensors(initialization_graph)
        return model

    def _instantiate_the_optimizer(self, model: nn.Module) -> None:
        model_parameters = list(model.parameters())
        self.optimizer = self.optimizer(model_parameters, lr=0.001, momentum=0.9)

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')

    @staticmethod
    def _get_current_batch_size(features: Any) -> int:
        return len(features)
