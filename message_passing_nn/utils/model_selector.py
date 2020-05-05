import logging

from torch import nn

from message_passing_nn.utils.models import models


class ModelSelector:
    def __init__(self) -> None:
        pass

    @staticmethod
    def load_model(model_selection: str) -> nn.Module:
        if model_selection in models:
            return models[model_selection]
        else:
            get_logger().info("The " + model_selection + " model is not available")
            raise Exception


def get_logger() -> logging.Logger:
    return logging.getLogger('message_passing_nn')
