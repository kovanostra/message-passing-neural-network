import logging
import os
from typing import Dict, List, Tuple

import itertools
import numpy as np
from torch.utils.data.dataloader import DataLoader

from message_passing_nn.data.data_preprocessor import DataPreprocessor
from message_passing_nn.model.trainer import Trainer
from message_passing_nn.repository.repository import Repository
from message_passing_nn.usecase import Usecase
from message_passing_nn.utils.saver import Saver


class GridSearch(Usecase):
    def __init__(self,
                 training_data_repository: Repository,
                 data_preprocessor: DataPreprocessor,
                 trainer: Trainer,
                 grid_search_dictionary: Dict,
                 saver: Saver,
                 cpu_multiprocessing: bool = False) -> None:
        self.repository = training_data_repository
        self.data_preprocessor = data_preprocessor
        self.trainer = trainer
        self.grid_search_dictionary = grid_search_dictionary
        self.saver = saver
        self.cpu_multiprocessing = cpu_multiprocessing

    def start(self) -> Dict:
        all_grid_search_configurations = self._get_all_grid_search_configurations()
        self.get_logger().info('Started Training')
        losses = {'training_loss': {},
                  'validation_loss': {},
                  'test_loss': {}}
        configuration_id = ''
        for configuration in all_grid_search_configurations:
            configuration_id, configuration_dictionary = self._get_configuration_dictionary(configuration)
            losses = self._search_configuration(configuration_id, configuration_dictionary, losses)
        self.saver.save_results(configuration_id, losses)
        self.get_logger().info('Finished Training')
        return losses

    def _search_configuration(self, configuration_id: str, configuration_dictionary: Dict, losses: Dict) -> Dict:
        training_data, validation_data, test_data, data_dimensions = self._prepare_dataset(configuration_dictionary)
        self.trainer.instantiate_attributes(data_dimensions, configuration_dictionary)
        losses = self._update_losses_with_configuration_id(configuration_dictionary, losses)
        validation_loss_max = np.inf
        cpu_cores_to_use = self._get_number_of_cpu_cores_to_use(training_data)
        self.get_logger().info('Starting training on ' + str(cpu_cores_to_use) + " CPU core(s)")
        for epoch in range(1, configuration_dictionary['epochs'] + 1):
            training_loss = self.trainer.do_train(training_data, epoch, cpu_cores_to_use)
            losses['training_loss'][configuration_dictionary["configuration_id"]].update({epoch: training_loss})
            if epoch % configuration_dictionary["validation_period"] == 0:
                validation_loss = self.trainer.do_evaluate(validation_data, epoch)
                losses['validation_loss'][configuration_dictionary["configuration_id"]].update(
                    {epoch: validation_loss})
                if validation_loss < validation_loss_max:
                    self.saver.save_model(epoch, configuration_id, self.trainer.model)
        test_loss = self.trainer.do_evaluate(test_data)
        losses['test_loss'][configuration_dictionary["configuration_id"]].update({"final_epoch": test_loss})
        return losses

    def _get_number_of_cpu_cores_to_use(self, training_data: DataLoader) -> int:
        if self.cpu_multiprocessing:
            if len(training_data) < os.cpu_count():
                cpu_cores_to_use = len(training_data)
            else:
                cpu_cores_to_use = os.cpu_count()
        else:
            cpu_cores_to_use = 1
        return cpu_cores_to_use

    @staticmethod
    def _update_losses_with_configuration_id(configuration_dictionary: Dict, losses: Dict) -> Dict:
        losses['training_loss'].update({configuration_dictionary["configuration_id"]: {}})
        losses['validation_loss'].update({configuration_dictionary["configuration_id"]: {}})
        losses['test_loss'].update({configuration_dictionary["configuration_id"]: {}})
        return losses

    @staticmethod
    def _get_configuration_dictionary(configuration: Tuple[Tuple]) -> Tuple[str, Dict]:
        configuration_dictionary = dict(((key, value) for key, value in configuration))
        configuration_id = 'configuration&id'
        for key, value in configuration_dictionary.items():
            configuration_id += "__" + "&".join([key, str(value)])
        configuration_dictionary.update({"configuration_id": configuration_id})
        return configuration_id, configuration_dictionary

    def _prepare_dataset(self, configuration_dictionary: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple]:
        raw_dataset = self.repository.get_all_data()
        self.get_logger().info("Calculating all neighbors for each node")
        dataset = self.data_preprocessor.find_all_node_neighbors(raw_dataset)
        training_data, validation_data, test_data = self.data_preprocessor \
            .train_validation_test_split(dataset,
                                         configuration_dictionary['batch_size'],
                                         configuration_dictionary['validation_split'],
                                         configuration_dictionary['test_split'])
        data_dimensions = self.data_preprocessor.extract_data_dimensions(dataset)
        return training_data, validation_data, test_data, data_dimensions

    def _get_all_grid_search_configurations(self) -> List[Tuple[Tuple]]:
        all_grid_search_configurations = []
        for key in self.grid_search_dictionary.keys():
            all_grid_search_configurations.append([(key, value) for value in self.grid_search_dictionary[key]])
        return list(itertools.product(*all_grid_search_configurations))

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
