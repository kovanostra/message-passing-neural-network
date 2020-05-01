import logging
import os
from datetime import datetime
from typing import Dict, List

import torch as to
from pandas import pandas as pd

from message_passing_nn.fixtures.filenames import *


class Saver:
    def __init__(self, model_directory: str, results_directory: str) -> None:
        self.model_directory = model_directory
        self.results_directory = results_directory

    def save_model(self, epoch: int, configuration_id: str, model: to.nn.Module) -> None:
        current_folder = self._join_path([self.model_directory, configuration_id])
        if not os.path.exists(current_folder):
            os.makedirs(current_folder)
        path_and_filename = self._join_path(
            [current_folder, self._join_strings([EPOCH, str(epoch), MODEL_STATE_DICTIONARY])])
        to.save(model.state_dict(), path_and_filename)
        self.get_logger().info("Saved model checkpoint in " + path_and_filename)

    def save_results(self, configuration_id: str, results: Dict) -> None:
        current_folder = self._join_path(
            [self.results_directory, configuration_id])
        if not os.path.exists(current_folder):
            os.makedirs(current_folder)
        results_dataframe = self._construct_dataframe_from_nested_dictionary(results)
        path_and_filename = self._join_path([current_folder,
                                             self._join_strings([datetime.now().strftime("%d-%b-%YT%H_%M"),
                                                                 RESULTS_CSV])])
        results_dataframe.to_csv(path_and_filename)
        self.get_logger().info("Saved results in " + path_and_filename)

    @staticmethod
    def _join_strings(fields: List) -> str:
        return "_".join(fields)

    @staticmethod
    def _construct_dataframe_from_nested_dictionary(results: Dict) -> pd.DataFrame:
        results_dataframe = pd.DataFrame.from_dict({(i, j): results[i][j]
                                                    for i in results.keys()
                                                    for j in results[i].keys()},
                                                   orient='index')
        return results_dataframe

    @staticmethod
    def _join_path(fields: List) -> str:
        return "/".join(fields)

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
