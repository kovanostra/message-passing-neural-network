from src.domain.loss_function_selector import LossFunctionSelector
from src.domain.optimizer_selector import OptimizerSelector
from src.repository.training_data_repository import TrainingDataRepository
from src.usecase.training import Training


class MessagePassingNN:
    def __init__(self, training, repository) -> None:
        self.training = training
        self.repository = repository

    def start(self):
        self.training.start(self.repository)


def create(dataset: str,
           epochs: int,
           loss_function_selection: str,
           optimizer_selection: str,
           data_path: str) -> MessagePassingNN:
    create_success_file()
    training_data_repository = TrainingDataRepository(data_path, dataset)
    loss_function = LossFunctionSelector(loss_function_selection).loss_function
    optimizer = OptimizerSelector(optimizer_selection).optimizer
    training = Training(epochs, loss_function, optimizer)
    return MessagePassingNN(training, training_data_repository)


def create_success_file():
    f = open("SUCCESS", "w")
    f.close()
