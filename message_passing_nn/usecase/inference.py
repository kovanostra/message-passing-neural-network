import logging
from typing import Tuple

from torch.utils.data.dataloader import DataLoader

from message_passing_nn.data import DataPreprocessor
from message_passing_nn.infrastructure.graph_dataset import GraphDataset
from message_passing_nn.model.inferencer import Inferencer
from message_passing_nn.model.loader import Loader
from message_passing_nn.usecase import Usecase
from message_passing_nn.utils import Saver


class Inference(Usecase):
    def __init__(self,
                 data_path: str,
                 data_preprocessor: DataPreprocessor,
                 loader: Loader,
                 inferencer: Inferencer,
                 saver: Saver,
                 test_mode: bool = False) -> None:
        self.data_path = data_path
        self.data_preprocessor = data_preprocessor
        self.loader = loader
        self.inferencer = inferencer
        self.saver = saver
        self.test_mode = test_mode

    def start(self) -> None:
        self.get_logger().info('Started Inference')
        configuration_id = ''
        inference_dataset, data_dimensions = self._prepare_dataset()
        model = self.loader.load_model(data_dimensions, self.saver.model_directory)
        outputs_labels_pairs = self.inferencer.do_inference(model, inference_dataset)
        self.saver.save_distance_maps(configuration_id, outputs_labels_pairs)
        self.get_logger().info('Finished Inference')

    def _prepare_dataset(self) -> Tuple[DataLoader, Tuple]:
        dataset = GraphDataset(self.data_path, test_mode=self.test_mode)
        inference_dataset = self.data_preprocessor.get_dataloader(dataset)
        data_dimensions = self.data_preprocessor.extract_data_dimensions(dataset)
        return inference_dataset, data_dimensions

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
