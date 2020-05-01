import logging
from typing import Dict, Any

import torch as to
from torch import nn
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from message_passing_nn.model.graph import Graph
from message_passing_nn.utils.loss_function_selector import LossFunctionSelector
from message_passing_nn.utils.optimizer_selector import OptimizerSelector


class ModelTrainer:
    def __init__(self, model: nn.Module, device: str) -> None:
        self.model = model
        self.loss_function = None
        self.optimizer = None
        self.device = device

    def instantiate_attributes(self, initialization_graph: Graph, configuration_dictionary: Dict) -> None:
        number_of_nodes = initialization_graph.number_of_nodes
        number_of_node_features = initialization_graph.number_of_node_features
        self.model = self.model.of(time_steps=configuration_dictionary['time_steps'],
                                   number_of_nodes=number_of_nodes,
                                   number_of_node_features=number_of_node_features,
                                   fully_connected_layer_input_size=number_of_nodes * number_of_node_features,
                                   fully_connected_layer_output_size=number_of_nodes ** 2,
                                   device=self.device)
        self.model.to(self.device)
        self.model.initialize_tensors(initialization_graph)
        self.loss_function = self._instantiate_the_loss_function(
            LossFunctionSelector.load_loss_function(configuration_dictionary['loss_function']))
        self.optimizer = self._instantiate_the_optimizer(
            OptimizerSelector.load_optimizer(configuration_dictionary['optimizer']))

    def do_train(self, training_data: DataLoader, epoch: int) -> float:
        training_loss = 0.0
        for features, labels in training_data:
            features, labels = features.to(self.device), labels.to(self.device)
            current_batch_size = self._get_current_batch_size(features)
            self.optimizer.zero_grad()
            outputs = self.model.forward(features, adjacency_matrix=labels, batch_size=current_batch_size)
            loss = self.loss_function(outputs, labels)
            training_loss += self._do_backpropagate(loss, training_loss)
        training_loss /= len(training_data)
        self.get_logger().info('[Iteration %d] training loss: %.3f' % (epoch, training_loss))
        return training_loss

    def do_evaluate(self, evaluation_data: DataLoader, epoch: int = None) -> float:
        with to.no_grad():
            evaluation_loss = 0.0
            for features_validation, labels_validation in evaluation_data:
                features_validation, labels_validation = features_validation.to(self.device), labels_validation.to(self.device)
                current_batch_size = self._get_current_batch_size(features_validation)
                outputs = self.model.forward(features_validation, labels_validation, current_batch_size)
                loss = self.loss_function(outputs, labels_validation)
                evaluation_loss += float(loss)
            evaluation_loss /= len(evaluation_data)
            if epoch is not None:
                self.get_logger().info('[Iteration %d] validation loss: %.3f' % (epoch, evaluation_loss))
            else:
                self.get_logger().info('Test loss: %.3f' % evaluation_loss)
        return evaluation_loss

    def _do_backpropagate(self, loss: to.Tensor, training_loss: float) -> float:
        loss.backward()
        self.optimizer.step()
        training_loss += loss.item()
        return training_loss

    @staticmethod
    def _instantiate_the_loss_function(loss_function: Module) -> Module:
        return loss_function()

    def _instantiate_the_optimizer(self, optimizer: Any) -> Optimizer:
        model_parameters = list(self.model.parameters())
        return optimizer(model_parameters, lr=0.001, momentum=0.9)

    @staticmethod
    def _get_current_batch_size(features: to.Tensor) -> int:
        return len(features)

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
