import logging
from typing import Dict, List, Tuple

import itertools
import numpy as np
from torch.utils.data.dataloader import DataLoader

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.model.graph import Graph
from message_passing_nn.repository.repository import Repository
from message_passing_nn.trainer.model_trainer import ModelTrainer
from message_passing_nn.utils.saver import Saver


class GridSearch:
    def __init__(self,
                 training_data_repository: Repository,
                 model_trainer: ModelTrainer,
                 grid_search_dictionary: Dict,
                 saver: Saver) -> None:
        self.repository = training_data_repository
        self.model_trainer = model_trainer
        self.grid_search_dictionary = grid_search_dictionary
        self.saver = saver

    def start(self) -> Dict:
        all_grid_search_configurations = self._get_all_grid_search_configurations()
        self.get_logger().info('Started Training')
        losses = {'training_loss': {},
                  'validation_loss': {},
                  'test_loss': {}}
        configuration_id = ''
        for configuration in all_grid_search_configurations:
            configuration_id, grid_search_configuration_dictionary = self._get_grid_search_configuration_dictionary(
                configuration)
            losses = self._search_configuration(configuration_id,
                                                grid_search_configuration_dictionary,
                                                losses)
        self.saver.save_results(configuration_id, losses)
        self.get_logger().info('Finished Training')
        return losses

    def _search_configuration(self,
                              configuration_id: str,
                              grid_search_configuration_dictionary: Dict,
                              losses: Dict) -> Dict:
        training_data, validation_data, test_data, initialization_graph = self._prepare_dataset(
            grid_search_configuration_dictionary)
        self.model_trainer.instantiate_attributes(initialization_graph, grid_search_configuration_dictionary)
        losses = self._update_losses_with_configuration_id(grid_search_configuration_dictionary, losses)
        validation_loss_max = np.inf
        for epoch in range(1, grid_search_configuration_dictionary['epochs'] + 1):
            training_loss = self.model_trainer.do_train(training_data, epoch)
            losses['training_loss'][grid_search_configuration_dictionary["configuration_id"]].update(
                {epoch: training_loss})
            if epoch % grid_search_configuration_dictionary["validation_period"] == 0:
                validation_loss = self.model_trainer.do_evaluate(validation_data, epoch)
                losses['validation_loss'][grid_search_configuration_dictionary["configuration_id"]].update(
                    {epoch: validation_loss})
                if validation_loss < validation_loss_max:
                    self.saver.save_model(epoch, configuration_id, self.model_trainer.model)
        test_loss = self.model_trainer.do_evaluate(test_data)
        losses['test_loss'][grid_search_configuration_dictionary["configuration_id"]].update(
            {"final_epoch": test_loss})
        return losses

    @staticmethod
    def _update_losses_with_configuration_id(configuration_dictionary: Dict, losses: Dict) -> Dict:
        losses['training_loss'].update({configuration_dictionary["configuration_id"]: {}})
        losses['validation_loss'].update({configuration_dictionary["configuration_id"]: {}})
        losses['test_loss'].update({configuration_dictionary["configuration_id"]: {}})
        return losses

    @staticmethod
    def _get_grid_search_configuration_dictionary(configuration: Tuple[Tuple]) -> Tuple[str, Dict]:
        grid_search_configuration_dictionary = dict(((key, value) for key, value in configuration))
        configuration_id = 'configuration_id'
        for key, value in grid_search_configuration_dictionary.items():
            configuration_id += "__" + "_".join([key, str(value)])
        grid_search_configuration_dictionary.update({"configuration_id": configuration_id})
        return configuration_id, grid_search_configuration_dictionary

    def _prepare_dataset(self, configuration_dictionary: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, Graph]:
        raw_dataset = self.repository.get_all_features_and_labels_from_separate_files()
        training_data, validation_data, test_data = DataPreprocessor \
            .train_validation_test_split(raw_dataset,
                                         configuration_dictionary['batch_size'],
                                         configuration_dictionary['validation_split'],
                                         configuration_dictionary['test_split'])
        initialization_graph = DataPreprocessor.extract_initialization_graph(raw_dataset)
        return training_data, validation_data, test_data, initialization_graph

    def _get_all_grid_search_configurations(self) -> List[Tuple[Tuple]]:
        all_grid_search_configurations = []
        for key in self.grid_search_dictionary.keys():
            all_grid_search_configurations.append([(key, value) for value in self.grid_search_dictionary[key]])
        return list(itertools.product(*all_grid_search_configurations))

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
