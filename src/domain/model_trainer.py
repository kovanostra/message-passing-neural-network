import logging
from typing import Any

import torch as to
from torch import nn

from src.domain.graph import Graph


class ModelTrainer:
    def __init__(self, model: nn.Module, loss_function: Any, optimizer: Any) -> None:
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

    def instantiate_model_and_optimizer(self, initialization_graph: Graph) -> None:
        number_of_nodes = initialization_graph.number_of_nodes
        number_of_node_features = initialization_graph.number_of_node_features
        self.model = self.model(time_steps=2,
                                number_of_nodes=number_of_nodes,
                                number_of_node_features=number_of_node_features,
                                fully_connected_layer_input_size=number_of_nodes * number_of_node_features,
                                fully_connected_layer_output_size=number_of_nodes ** 2)
        self.model.initialize_tensors(initialization_graph)
        self.optimizer = self._instantiate_the_optimizer(self.optimizer)

    def do_train(self, epoch: int, training_data: Any) -> Any:
        training_loss = 0.0
        for features, labels in training_data:
            current_batch_size = self._get_current_batch_size(features)
            self.optimizer.zero_grad()
            outputs = self.model.forward(features, adjacency_matrix=labels, batch_size=current_batch_size)
            loss = self.loss_function(outputs, labels)
            training_loss += self._do_backpropagate(epoch, loss, training_loss)
        training_loss /= len(training_data)
        return training_loss

    def do_evaluate(self, evaluation_data: Any, epoch: int = None) -> float:
        with to.no_grad():
            evaluation_loss = 0.0
            for features_validation, labels_validation in evaluation_data:
                self.model.eval()
                current_batch_size = self._get_current_batch_size(features_validation)
                outputs = self.model.forward(features_validation, labels_validation, current_batch_size)
                loss = self.loss_function(outputs, labels_validation)
                evaluation_loss += float(loss)
            evaluation_loss /= len(evaluation_data)
            if epoch is not None:
                self.get_logger().info('[%d] validation loss: %.3f' % (epoch + 1, evaluation_loss))
            else:
                self.get_logger().info('Test loss: %.3f' % evaluation_loss)
        return evaluation_loss

    def _do_backpropagate(self, epoch: int, loss: Any, training_loss: float) -> float:
        loss.backward()
        self.optimizer.step()
        training_loss += loss.item()
        self.get_logger().info('[%d] training loss: %.3f' % (epoch + 1, training_loss))
        return training_loss

    def _instantiate_the_optimizer(self, optimizer: Any) -> Any:
        model_parameters = list(self.model.parameters())
        return optimizer(model_parameters, lr=0.001, momentum=0.9)

    @staticmethod
    def _get_current_batch_size(features: Any) -> int:
        return len(features)

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
