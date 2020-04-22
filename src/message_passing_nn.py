from src.domain.loss_function_selector import LossFunctionSelector
from src.domain.optimizer_selector import OptimizerSelector
from src.repository.interface.repository import Repository
from src.repository.training_data_repository import TrainingDataRepository
from src.usecase.training import Training


class MessagePassingNN:
    def __init__(self, training: Training, repository: Repository, batch_size: int) -> None:
        self.training = training
        self.repository = repository
        self.batch_size = batch_size

    def start(self):
        self.training.start(self.repository, self.batch_size)


def create(dataset: str,
           epochs: int,
           loss_function_selection: str,
           optimizer_selection: str,
           data_path: str,
           batch_size: int) -> MessagePassingNN:
    create_success_file()
    training_data_repository = TrainingDataRepository(data_path, dataset)
    loss_function = LossFunctionSelector(loss_function_selection).loss_function
    optimizer = OptimizerSelector(optimizer_selection).optimizer
    training = Training(epochs, loss_function, optimizer)
    return MessagePassingNN(training, training_data_repository, batch_size)


def create_success_file():
    f = open("SUCCESS", "w")
    f.close()
