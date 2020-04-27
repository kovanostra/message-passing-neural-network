import logging
from typing import Any

from src.fixtures.loss_functions import loss_functions


class LossFunctionSelector:
    def __init__(self, loss_function_selection: str) -> None:
        self.loss_function = self.loss_function_loader(loss_function_selection)

    def loss_function_loader(self, loss_function_selection: str) -> Any:
        if loss_function_selection in loss_functions:
            return loss_functions[loss_function_selection]
        else:
            self.get_logger().info("The " + loss_function_selection + " is not available")
            raise Exception

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
