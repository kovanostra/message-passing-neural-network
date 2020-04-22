import logging
from typing import Any, List

import torch
from torch.utils.data import DataLoader

from src.domain.fully_connected_layer import FullyConnectedLayer
from src.domain.graph import Graph
from src.domain.graph_dataset import GraphDataset
from src.domain.graph_encoder import GraphEncoder
from src.repository.interface.repository import Repository


class Training:
    def __init__(self, epochs: int, loss_function: Any, optimizer: Any) -> None:
        self.epochs = epochs
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.running_loss = 0.0

    def start(self, repository: Repository, batch_size: int):
        training_data, initialization_graph = self._prepare_dataset(repository, batch_size)
        graph_encoder = self._instantiate_graph_encoder(initialization_graph)
        fully_connected_layer = self._instantiate_fully_connected_layer(initialization_graph, batch_size)
        self._instantiate_the_optimizer(graph_encoder, fully_connected_layer)
        self.get_logger().info('Started Training')
        for epoch in range(self.epochs):
            self.running_loss = 0.0
            for features, labels in training_data:
                current_batch_size = self._get_current_batch_size(features)
                self.optimizer.zero_grad()
                with torch.autograd.set_detect_anomaly(True):
                    loss = self._make_a_forward_pass(graph_encoder,
                                                     fully_connected_layer,
                                                     features,
                                                     labels,
                                                     current_batch_size)
                    self.running_loss = self._backpropagate_the_errors(epoch, loss, self.running_loss)
        self.get_logger().info('Finished Training')

    def _prepare_dataset(self, repository: Repository, batch_size: int) -> Any:
        raw_training_data = repository.get_all_features_and_labels_from_separate_files()
        graph_dataset = GraphDataset(raw_training_data)
        training_data = DataLoader(graph_dataset, batch_size)
        return training_data, self._extract_initialization_graph(raw_training_data)

    @staticmethod
    def _instantiate_graph_encoder(initialization_graph: Graph) -> GraphEncoder:
        graph_encoder = GraphEncoder(time_steps=2,
                                     number_of_nodes=initialization_graph.number_of_nodes,
                                     number_of_node_features=initialization_graph.number_of_node_features)
        graph_encoder.initialize_tensors(initialization_graph)
        return graph_encoder

    @staticmethod
    def _instantiate_fully_connected_layer(initialization_graph: Graph, batch_size: int) -> Any:
        fully_connected_input_size = batch_size * \
                                     initialization_graph.number_of_nodes * \
                                     initialization_graph.number_of_node_features
        fully_connected_output_size = initialization_graph.number_of_nodes ** 2
        fully_connected_layer = FullyConnectedLayer(fully_connected_input_size, fully_connected_output_size)
        return fully_connected_layer

    def _instantiate_the_optimizer(self, graph_encoder: GraphEncoder, fully_connected_layer: Any) -> None:
        model_parameters = list(graph_encoder.parameters()) + list(fully_connected_layer.parameters())
        self.optimizer = self.optimizer(model_parameters, lr=0.001, momentum=0.9)

    def _make_a_forward_pass(self,
                             graph_encoder: GraphEncoder,
                             fully_connected_layer: FullyConnectedLayer,
                             features: Any,
                             labels: Any,
                             current_batch_size: int) -> Any:
        graph_outputs = graph_encoder.forward(features, adjacency_matrix=labels, batch_size=current_batch_size)
        graph_outputs_flattened = self._flatten(graph_outputs, current_batch_size)
        outputs = fully_connected_layer(graph_outputs_flattened)
        loss = self.loss_function(outputs, labels)
        return loss

    def _backpropagate_the_errors(self, epoch: int, loss: Any, running_loss: float) -> float:
        loss.backward()
        self.optimizer.step()
        running_loss += loss.item()
        self.get_logger().info('[%d] loss: %.3f' % (epoch + 1, running_loss))
        return running_loss

    @staticmethod
    def _extract_labels_from_graph(graph: Graph) -> Any:
        return graph.adjacency_matrix.float().view(-1)

    @staticmethod
    def _flatten(tensors: List[Any], batch_size: int) -> Any:
        return tensors.view(batch_size, -1)

    @staticmethod
    def _extract_initialization_graph(training_data: Any) -> Graph:
        return Graph(training_data[0][1], training_data[0][0])

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')

    @staticmethod
    def _get_current_batch_size(features: Any) -> int:
        return len(features)
