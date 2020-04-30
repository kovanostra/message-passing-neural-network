import logging

from src.domain.graph_encoder import GraphEncoder
from src.domain.grid_search_parameters_parser import GridSearchParametersParser
from src.domain.model_trainer import ModelTrainer
from src.domain.saver import Saver
from src.repository.training_data_repository import TrainingDataRepository
from src.usecase.grid_search import GridSearch


class MessagePassingNN:
    def __init__(self, grid_search: GridSearch) -> None:
        self.grid_search = grid_search

    def start(self):
        try:
            self.grid_search.start()
        except Exception:
            get_logger().exception("message")


def create(dataset_name: str,
           data_directory: str,
           model_directory: str,
           results_directory: str,
           device: str,
           epochs: str,
           loss_function_selection: str,
           optimizer_selection: str,
           batch_size: str,
           validation_split: str,
           test_split: str,
           time_steps: str,
           validation_period: str) -> MessagePassingNN:
    create_success_file()
    grid_search_dictionary = GridSearchParametersParser().get_grid_search_dictionary(epochs,
                                                                                     loss_function_selection,
                                                                                     optimizer_selection,
                                                                                     batch_size,
                                                                                     validation_split,
                                                                                     test_split,
                                                                                     time_steps,
                                                                                     validation_period)
    training_data_repository = TrainingDataRepository(data_directory, dataset_name)
    model_trainer = ModelTrainer(GraphEncoder, device)
    saver = Saver(model_directory, results_directory)
    grid_search = GridSearch(training_data_repository, model_trainer, grid_search_dictionary, saver)
    return MessagePassingNN(grid_search)


def get_logger() -> logging.Logger:
    return logging.getLogger('message_passing_nn')


def create_success_file():
    f = open("SUCCESS", "w")
    f.close()
