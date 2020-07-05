import logging
from typing import Any

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.model.inferencer import Inferencer
from message_passing_nn.model.loader import Loader
from message_passing_nn.model.trainer import Trainer
from message_passing_nn.repository.file_system_repository import FileSystemRepository
from message_passing_nn.usecase.grid_search import GridSearch
from message_passing_nn.usecase.inference import Inference
from message_passing_nn.utils.grid_search_parameters_parser import GridSearchParametersParser
from message_passing_nn.utils.saver import Saver
from message_passing_nn.usecase import Usecase


class MessagePassingNN:
    def __init__(self, usecase: Usecase) -> None:
        self.usecase = usecase

    def start(self):
        try:
            self.usecase.start()
        except Exception:
            get_logger().exception("message")


def create_grid_search(dataset_name: str,
                       data_directory: str,
                       model_directory: str,
                       results_directory: str,
                       model: str,
                       device: str,
                       epochs: str,
                       loss_function_selection: str,
                       optimizer_selection: str,
                       batch_size: str,
                       maximum_number_of_features: str,
                       maximum_number_of_nodes: str,
                       validation_split: str,
                       test_split: str,
                       time_steps: str,
                       validation_period: str) -> MessagePassingNN:
    grid_search_dictionary = GridSearchParametersParser().get_grid_search_dictionary(model,
                                                                                     epochs,
                                                                                     loss_function_selection,
                                                                                     optimizer_selection,
                                                                                     batch_size,
                                                                                     maximum_number_of_features,
                                                                                     maximum_number_of_nodes,
                                                                                     validation_split,
                                                                                     test_split,
                                                                                     time_steps,
                                                                                     validation_period)
    file_system_repository = FileSystemRepository(data_directory, dataset_name)
    data_preprocessor = DataPreprocessor()
    trainer = Trainer(data_preprocessor, device)
    saver = Saver(model_directory, results_directory)
    grid_search = GridSearch(file_system_repository, data_preprocessor, trainer, grid_search_dictionary, saver)
    return MessagePassingNN(grid_search)


def create_inference(dataset_name: str,
                     data_directory: str,
                     model_directory: str,
                     results_directory: str,
                     model: str,
                     device: str) -> MessagePassingNN:
    file_system_repository = FileSystemRepository(data_directory, dataset_name)
    data_preprocessor = DataPreprocessor()
    model_loader = Loader(model)
    model_inferencer = Inferencer(device)
    saver = Saver(model_directory, results_directory)
    inference = Inference(file_system_repository, data_preprocessor, model_loader, model_inferencer, saver)
    return MessagePassingNN(inference)


def get_logger() -> logging.Logger:
    return logging.getLogger('message_passing_nn')
