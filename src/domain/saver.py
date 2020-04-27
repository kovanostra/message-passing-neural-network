import os
from datetime import datetime
from typing import Any, Dict

from pandas import pandas as pd

from src.fixtures.filenames import RESULTS_CSV


class Saver:
    def __init__(self, results_path: str) -> None:
        self.results_path = results_path

    def save_model(self, model: Any) -> None:
        pass

    def save_results(self, results: Dict) -> None:
        filename = datetime.now().strftime("%d-%b-%YT%H:%M:%S")
        if not os.path.exists(self.results_path + filename):
            os.makedirs('my_folder')
        results_dataframe = pd.DataFrame.from_dict(results)
        results_dataframe.to_csv(self.results_path + filename + RESULTS_CSV)
