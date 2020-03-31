from src.repository.training_data_repository import TrainingDataRepository
from src.usecase.training import Training


class MessagePassingNN:
    def __init__(self, training, repository) -> None:
        self.training = training
        self.repository = repository

    def start(self):
        self.training.start(self.repository)


def create(dataset: str, epochs: int, loss_function: str, optimizer: str, datapath: str) -> MessagePassingNN:
    create_success_file()
    training_data_repository = TrainingDataRepository(dataset, datapath)
    training = Training(epochs, loss_function, optimizer)
    return MessagePassingNN(training, training_data_repository)


def create_success_file():
    f = open("SUCCESS", "w")
    f.close()
