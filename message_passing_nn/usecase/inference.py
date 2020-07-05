import logging

from torch.utils.data.dataloader import DataLoader

from message_passing_nn.data import DataPreprocessor
from message_passing_nn.repository import Repository
from message_passing_nn.utils import Saver
from message_passing_nn.model.loader import Loader

from message_passing_nn.model.inferencer import Inferencer


class Inference:
    def __init__(self,
                 training_data_repository: Repository,
                 data_preprocessor: DataPreprocessor,
                 loader: Loader,
                 inferencer: Inferencer,
                 saver: Saver) -> None:
        self.repository = training_data_repository
        self.data_preprocessor = data_preprocessor
        self.loader = loader
        self.inferencer = inferencer
        self.saver = saver

    def start(self) -> None:
        self.get_logger().info('Started Inference')
        configuration_id = ''
        model = self.loader.load_model()
        inference_dataset = self._prepare_dataset()
        outputs, labels = self.inferencer.do_inference(model, inference_dataset)
        outputs_distance_map, labels_distance_map = self.data_preprocessor.get_distance_maps(outputs, labels)
        self.saver.save_distance_maps(configuration_id, outputs_distance_map, labels_distance_map)
        self.get_logger().info('Finished Inference')

    def _prepare_dataset(self) -> DataLoader:
        raw_dataset = self.repository.get_all_data()
        equalized_dataset = self.data_preprocessor.equalize_dataset_dimensions(raw_dataset)
        inference_dataset = self.data_preprocessor.get_dataloader(equalized_dataset)
        return inference_dataset

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
