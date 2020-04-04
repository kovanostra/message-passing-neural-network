from typing import Any

import torch as to

from src.domain.loss_function_selector import LossFunctionSelector
from src.domain.optimizer_selector import OptimizerSelector
from src.repository.interface.repository import Repository
from src.repository.training_data_repository import TrainingDataRepository
from src.usecase.training import Training


class MessagePassingNN:
    def __init__(self, training: Training, repository: Repository, device: Any, use_all_gpus: str) -> None:
        self.training = training
        self.repository = repository
        self.device = device
        self.use_all_gpus = multiple_gpus_available(use_all_gpus)

    def start(self):
        self.training.start(self.repository, self.device, self.use_all_gpus)


def create(dataset: str,
           epochs: int,
           loss_function_selection: str,
           optimizer_selection: str,
           data_path: str,
           enable_gpu: str,
           which_gpu: str,
           use_all_gpus: str) -> MessagePassingNN:
    create_success_file()
    device = setup_gpu(enable_gpu, which_gpu)
    training_data_repository = TrainingDataRepository(data_path, dataset)
    loss_function = LossFunctionSelector(loss_function_selection).loss_function
    optimizer = OptimizerSelector(optimizer_selection).optimizer
    training = Training(epochs, loss_function, optimizer)
    return MessagePassingNN(training, training_data_repository, device, use_all_gpus)


def setup_gpu(enable_gpu: str, which_gpu: str) -> Any:
    if to_boolean(enable_gpu):
        gpu_to_use = which_gpu if which_gpu else 'cuda'
        device = to.device(gpu_to_use if to.cuda.is_available() else "cpu")
    else:
        device = to.device("cpu")
    return device


def multiple_gpus_available(use_all_gpus: str) -> bool:
    if to_boolean(use_all_gpus):
        return to.cuda.device_count() > 1
    else:
        return False


def to_boolean(field: str) -> bool:
    return eval(field.capitalize())


def create_success_file():
    f = open("SUCCESS", "w")
    f.close()
