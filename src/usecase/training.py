import logging

from torch import nn, optim

from domain.graph_preprocessor import GraphPreprocessor
from src.domain.graph_encoder import GraphEncoder
from src.repository.interface.repository import Repository


class Training:
    def __init__(self, epochs, loss_function, optimizer) -> None:
        self.epochs = epochs
        self.loss_function = loss_function
        self.optimizer = optimizer

    def start(self, repository: Repository):
        training_data = repository.get_all_features_and_labels_from_separate_files()
        graph_preprocessor = GraphPreprocessor()
        training_data_in_batches = graph_preprocessor.preprocess(training_data, batches=5)
        for epoch in range(self.epochs):
            for batch in training_data_in_batches:
                for graph in batch:
                    graph_encoder = GraphEncoder()
                    graph_encoder.initialize_tensors(graph)
                    self.loss_function = nn.MSELoss()
                    self.optimizer = optim.SGD(graph_encoder.parameters(), lr=0.001, momentum=0.9)
                    running_loss = 0.0

                    self.optimizer.zero_grad()

                    outputs = graph_encoder.forward(graph)
                    loss = self.loss_function(outputs, graph.adjacency_matrix.view(-1))
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    self.get_logger().info('[%d] loss: %.3f' % (epoch + 1, running_loss))

        self.get_logger().info('Finished Training')
        return running_loss

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
