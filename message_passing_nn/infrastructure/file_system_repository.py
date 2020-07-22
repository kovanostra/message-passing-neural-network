import logging
import pickle

import torch as to


class FileSystemRepository:
    def __init__(self, data_directory: str, dataset: str) -> None:
        super().__init__()
        self.data_directory = data_directory + dataset + '/'
        self.test_mode = False

    def save(self, filename: str, data_to_save: to.Tensor) -> None:
        with open(self.data_directory + filename, 'wb') as file:
            pickle.dump(data_to_save, file)

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
