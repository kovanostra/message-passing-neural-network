from src.repository.training_data_repository import TrainingDataRepository
from src.usecase.train import Train


class MessagePassingNN:
    def __init__(self, train, repository) -> None:
        self.train = train
        self.repository = repository

    def start(self):
        self.train.start(self.repository)


def create(epochs: int, loss_function: str, optimizer: str) -> MessagePassingNN:
    create_success_file()
    training_data_repository = TrainingDataRepository()
    train = Train(epochs, loss_function, optimizer)
    return MessagePassingNN(train, training_data_repository)


def create_success_file():
    f = open("SUCCESS", "w")
    f.close()
