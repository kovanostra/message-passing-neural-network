import logging
from multiprocessing import get_context
from typing import Dict, Any, Tuple

import numpy as np

import torch as to
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from message_passing_nn.data.preprocessor import Preprocessor
from message_passing_nn.utils.loss_function_selector import LossFunctionSelector
from message_passing_nn.utils.model_selector import ModelSelector
from message_passing_nn.utils.optimizer_selector import OptimizerSelector


class Trainer:
    def __init__(self, preprocessor: Preprocessor, device: str, normalize: bool = False) -> None:
        self.preprocessor = preprocessor
        self.device = device
        self.normalize = normalize
        self.model = None
        self.loss_function = None
        self.optimizer = None

    def instantiate_attributes(self,
                               data_dimensions: Tuple,
                               configuration_dictionary: Dict) -> None:
        node_features_size, adjacency_matrix_size, labels_size = data_dimensions
        number_of_nodes = adjacency_matrix_size[0]
        number_of_node_features = node_features_size[1]
        fully_connected_layer_output_size = labels_size[0]
        self.model = ModelSelector.load_model(configuration_dictionary['model'])
        self.model = self.model(time_steps=configuration_dictionary['time_steps'],
                                number_of_nodes=number_of_nodes,
                                number_of_node_features=number_of_node_features,
                                fully_connected_layer_input_size=number_of_nodes * number_of_node_features,
                                fully_connected_layer_output_size=fully_connected_layer_output_size)
        self.get_logger().info('Loaded the ' + configuration_dictionary['model'] +
                               ' model. Model size: ' + self.model.get_model_size() + ' MB')
        self.model.to(self.device)
        self.loss_function = self._instantiate_the_loss_function(
            LossFunctionSelector.load_loss_function(configuration_dictionary['loss_function']))
        self.get_logger().info('Loss function: ' + configuration_dictionary['loss_function'])
        self.optimizer = self._instantiate_the_optimizer(
            OptimizerSelector.load_optimizer(configuration_dictionary['optimizer']))
        self.get_logger().info('Optimizer: ' + configuration_dictionary['optimizer'])

    def do_train(self, training_data: DataLoader, epoch: int, cpu_cores_to_use: int = 1) -> float:
        training_loss = 0.0
        if cpu_cores_to_use > 1 and self.device == 'cpu':
            with get_context("spawn").Pool(cpu_cores_to_use) as pool:
                training_loss += np.average(pool.map(self._do_train_batch, training_data))
        else:
            training_loss += np.average(list(map(self._do_train_batch, training_data)))
        self.get_logger().info('[Iteration %d] training loss: %.6f' % (epoch, training_loss))
        return training_loss

    def _do_train_batch(self, training_data: DataLoader) -> float:
        features, labels = training_data
        node_features, adjacency_matrix = features
        node_features, adjacency_matrix, labels = node_features.to(self.device), \
                                                  adjacency_matrix.to(self.device), \
                                                  labels.to(self.device)
        current_batch_size = self._get_current_batch_size(labels)
        if self.normalize:
            node_features = self.preprocessor.normalize(node_features, self.device)
            labels = self.preprocessor.normalize(labels, self.device)
        self.optimizer.zero_grad()
        outputs = self.model.forward(node_features, adjacency_matrix, batch_size=current_batch_size)
        loss = self.loss_function(outputs, labels)
        self._do_backpropagate(loss)
        return loss.item()

    def do_evaluate(self, evaluation_data: DataLoader, epoch: int = None) -> float:
        with to.no_grad():
            evaluation_loss = []
            if len(evaluation_data):
                for features_validation, labels_validation in evaluation_data:
                    node_features, adjacency_matrix = features_validation
                    node_features, adjacency_matrix, labels_validation = node_features.to(self.device), \
                                                                         adjacency_matrix.to(self.device), \
                                                                         labels_validation.to(self.device)
                    if self.normalize:
                        node_features = self.preprocessor.normalize(node_features, self.device)
                        labels_validation = self.preprocessor.normalize(labels_validation, self.device)
                    current_batch_size = self._get_current_batch_size(labels_validation)
                    outputs = self.model.forward(node_features, adjacency_matrix, current_batch_size)
                    loss = self.loss_function(outputs, labels_validation)
                    evaluation_loss.append(float(loss))
                evaluation_loss = np.average(evaluation_loss)
                if epoch is not None:
                    self.get_logger().info('[Iteration %d] validation loss: %.6f' % (epoch, evaluation_loss))
                else:
                    self.get_logger().info('Test loss: %.6f' % evaluation_loss)
            else:
                self.get_logger().warning('No evaluation data found!')
        return evaluation_loss

    def _do_backpropagate(self, loss: to.Tensor) -> None:
        loss.backward()
        self.optimizer.step()

    @staticmethod
    def _instantiate_the_loss_function(loss_function: Module) -> Module:
        return loss_function()

    def _instantiate_the_optimizer(self, optimizer: Any) -> Optimizer:
        model_parameters = list(self.model.parameters())
        try:
            optimizer = optimizer(model_parameters, lr=0.001, momentum=0.9)
        except:
            optimizer = optimizer(model_parameters, lr=0.001)
        return optimizer

    @staticmethod
    def _get_current_batch_size(features: to.Tensor) -> int:
        return len(features)

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
