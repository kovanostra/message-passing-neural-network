import torch as to

from torch import nn
from typing import Dict, Tuple

from message_passing_nn.utils import ModelSelector


class Loader:
    def __init__(self, model: str) -> None:
        self.model = ModelSelector.load_model(model)

    def load_model(self, data_dimensions: Tuple, path_to_model: str) -> nn.Module:
        model_parameters = self._get_model_parameters_from_path(path_to_model)
        node_features_size, adjacency_matrix_size, labels_size = data_dimensions
        number_of_nodes = adjacency_matrix_size[0]
        number_of_node_features = node_features_size[1]
        fully_connected_layer_output_size = labels_size[0]
        self.model = self.model(time_steps=model_parameters['time_steps'],
                                number_of_nodes=number_of_nodes,
                                number_of_node_features=number_of_node_features,
                                fully_connected_layer_input_size=number_of_nodes * number_of_node_features,
                                fully_connected_layer_output_size=fully_connected_layer_output_size)
        self.model.load_state_dict(to.load(path_to_model))
        self.model.eval()
        return self.model

    @staticmethod
    def _get_model_parameters_from_path(path_to_model: str) -> Dict:
        model_configuration = path_to_model.split("/")[-2].split("__")
        model_parameters = {}
        for model_parameter in model_configuration:
            key, value = model_parameter.split("_")[0], model_parameter.split("_")[1]
            model_parameters.update({key: value})
        return model_parameters
