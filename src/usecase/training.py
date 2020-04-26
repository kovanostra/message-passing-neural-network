import logging
from typing import Any, Tuple

from src.domain.data_preprocessor import DataPreprocessor
from src.domain.model_trainer import ModelTrainer
from src.repository.interface.repository import Repository


class Training:
    def __init__(self, training_data_repository: Repository, model_trainer: ModelTrainer, epochs: int) -> None:
        self.repository = training_data_repository
        self.model_trainer = model_trainer
        self.epochs = epochs

    def start(self, batch_size: int, validation_split: float, test_split: float) -> Tuple[float, float, float]:
        training_data, validation_data, test_data, initialization_graph = self._prepare_dataset(batch_size,
                                                                                                validation_split,
                                                                                                test_split)
        self.model_trainer.instantiate_model_and_optimizer(initialization_graph)
        self.get_logger().info('Started Training')
        training_loss, validation_loss, test_loss = (0.0, 0.0, 0.0)
        for epoch in range(self.epochs):
            training_loss = self.model_trainer.do_train(epoch, training_data)
            if epoch % 10 == 0:
                validation_loss = self.model_trainer.do_evaluate(validation_data, epoch)
        test_loss = self.model_trainer.do_evaluate(test_data)
        self.get_logger().info('Finished Training')
        return training_loss, validation_loss, test_loss

    def _prepare_dataset(self, batch_size: int, validation_split: float, test_split: float) -> Any:
        raw_dataset = self.repository.get_all_features_and_labels_from_separate_files()
        training_data, validation_data, test_data = DataPreprocessor.train_validation_test_split(raw_dataset,
                                                                                                 batch_size,
                                                                                                 validation_split,
                                                                                                 test_split)
        initialization_graph = DataPreprocessor.extract_initialization_graph(raw_dataset)
        return training_data, validation_data, test_data, initialization_graph

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
