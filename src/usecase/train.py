from torch import nn, optim

from src.domain.graph_encoder import GraphEncoder
from src.repository.interface.repository import Repository


class Train:
    def __init__(self, epochs, loss_function, optimizer) -> None:
        self.epochs = epochs
        self.loss_function = loss_function
        self.optimizer = optimizer

    def start(self, repository: Repository):
        training_dataset = repository.get_all()[0]
        graph_encoder = GraphEncoder(training_dataset)
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.SGD(graph_encoder.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(self.epochs):
            running_loss = 0.0

            self.optimizer.zero_grad()

            outputs = graph_encoder.forward(training_dataset)
            loss = self.loss_function(outputs, training_dataset.node_features)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            print('[%d] loss: %.3f' % (epoch + 1, running_loss))

        print('Finished Training')
        return running_loss