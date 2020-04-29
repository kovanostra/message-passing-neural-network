import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from pandas import pandas as pd

from src.fixtures.filenames import RESULTS_CSV


class Saver:
    def __init__(self, results_directory: str) -> None:
        self.results_directory = results_directory

    def save_model(self, model: Any) -> None:
        pass

    def save_results(self, results: Dict) -> None:
        current_folder = self._join_path([self.results_directory, datetime.now().strftime("%d-%b-%YT%H_%M")])
        if not os.path.exists(current_folder):
            os.makedirs(current_folder)
        results_dataframe = self._construct_dataframe_from_nested_dictionary(results)
        path_and_filename = self._join_path([current_folder, RESULTS_CSV])
        results_dataframe.to_csv(path_and_filename)
        self.get_logger().info("Saved results in " + path_and_filename)

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
