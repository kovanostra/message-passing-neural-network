import logging
from typing import Any

from torch import nn, optim

from src.domain.fully_connected_layer import FullyConnectedLayer
from src.domain.graph import Graph
from src.domain.graph_encoder import GraphEncoder
from src.domain.graph_preprocessor import GraphPreprocessor
from src.repository.interface.repository import Repository


class Training:
    def __init__(self, epochs: int, loss_function: Any, optimizer: Any) -> None:
        self.epochs = epochs
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.running_loss = 0.0

    def start(self, repository: Repository):
        training_data_in_batches = self._prepare_dataset(repository, batches=5)
        graph_encoder = self._create_graph_encoder(training_data_in_batches)
        fully_connected_layer = self._create_fully_connected_layer(training_data_in_batches)
        self._set_the_loss_function_and_optimizer(fully_connected_layer, graph_encoder)
        self.get_logger().info('Started Training')
        for epoch in range(self.epochs):
            self._feed_batches(epoch, fully_connected_layer, graph_encoder, training_data_in_batches)
        self.get_logger().info('Finished Training')

    @staticmethod
    def _prepare_dataset(repository: Repository, batches: int) -> Any:
        training_data = repository.get_all_features_and_labels_from_separate_files()
        graph_preprocessor = GraphPreprocessor()
        training_data_in_batches = graph_preprocessor.preprocess(training_data, batches)
        return training_data_in_batches

    def _create_graph_encoder(self, training_data_in_batches: Any) -> GraphEncoder:
        initialization_graph = self._extract_initialization_graph(training_data_in_batches)
        graph_encoder = GraphEncoder()
        graph_encoder.initialize_tensors(initialization_graph)
        return graph_encoder

    def _create_fully_connected_layer(self, training_data_in_batches: Any) -> Any:
        initialization_graph = self._extract_initialization_graph(training_data_in_batches)
        fully_connected_input_size = initialization_graph.number_of_nodes * initialization_graph.number_of_node_features
        fully_connected_output_size = initialization_graph.number_of_nodes ** 2
        fully_connected_layer = FullyConnectedLayer(fully_connected_input_size, fully_connected_output_size)
        return fully_connected_layer

    def _set_the_loss_function_and_optimizer(self, graph_encoder: GraphEncoder, fully_connected_layer: Any) -> None:
        self.loss_function = nn.MSELoss()
        model_parameters = list(graph_encoder.parameters()) + list(fully_connected_layer.parameters())
        self.optimizer = optim.SGD(model_parameters, lr=0.001, momentum=0.9)

    def _feed_batches(self,
                      epoch: int,
                      fully_connected_layer: Any,
                      graph_encoder: GraphEncoder,
                      training_data_in_batches: Any) -> None:
        for batch in training_data_in_batches:
            self.running_loss = self._train_batch(batch, epoch, fully_connected_layer, graph_encoder)

    def _train_batch(self,
                     batch: Any,
                     epoch: int,
                     fully_connected_layer: Any,
                     graph_encoder: GraphEncoder,
                     running_loss=0.0) -> float:
        for graph in batch:
            self.optimizer.zero_grad()
            loss = self._make_a_forward_pass(fully_connected_layer, graph, graph_encoder)
            running_loss = self._backpropagate_the_errors(epoch, loss, running_loss)
        return running_loss

    def _make_a_forward_pass(self,
                             fully_connected_layer: Any,
                             graph: Graph,
                             graph_encoder: GraphEncoder) -> Any:
        graph_outputs = graph_encoder.forward(graph)
        outputs = fully_connected_layer(graph_outputs.view(-1).float())
        labels = self._extract_labels_from_graph(graph)
        loss = self.loss_function(outputs, labels)
        return loss

    @staticmethod
    def _extract_labels_from_graph(graph: Graph) -> Any:
        return graph.adjacency_matrix.float().view(-1)

    def _backpropagate_the_errors(self, epoch: int, loss: Any, running_loss: float) -> float:
        loss.backward()
        self.optimizer.step()
        running_loss += loss.item()
        self.get_logger().info('[%d] loss: %.3f' % (epoch + 1, running_loss))
        return running_loss

    @staticmethod
    def _extract_initialization_graph(training_data_in_batches: Any) -> Graph:
        return training_data_in_batches[0][0]

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
