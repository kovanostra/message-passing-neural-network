from src.domain.graph_encoder import GraphEncoder
from src.domain.loss_function_selector import LossFunctionSelector
from src.domain.model_trainer import ModelTrainer
from src.domain.optimizer_selector import OptimizerSelector
from src.repository.training_data_repository import TrainingDataRepository
from src.usecase.training import Training


class MessagePassingNN:
    def __init__(self, training: Training, batch_size: int, validation_split: float, test_split: float) -> None:
        self.training = training
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split

    def start(self):
        self.training.start(self.batch_size, self.validation_split, self.test_split)


def create(dataset: str,
           epochs: int,
           loss_function_selection: str,
           optimizer_selection: str,
           data_path: str,
           batch_size: int,
           validation_split: float,
           test_split: float) -> MessagePassingNN:
    create_success_file()
    training_data_repository = TrainingDataRepository(data_path, dataset)
    loss_function = LossFunctionSelector(loss_function_selection).loss_function
    optimizer = OptimizerSelector(optimizer_selection).optimizer
    model_trainer = ModelTrainer(GraphEncoder, loss_function, optimizer)
    training = Training(training_data_repository, model_trainer, epochs)
    return MessagePassingNN(training, batch_size, validation_split, test_split)


def create_success_file():
    f = open("SUCCESS", "w")
    f.close()
