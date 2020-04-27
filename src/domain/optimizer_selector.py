import logging
from typing import Any

from src.fixtures.optimizers import optimizers


class OptimizerSelector:
    def __init__(self, optimizer_selection: str) -> None:
        self.optimizer = self.optimizer_loader(optimizer_selection)

    def optimizer_loader(self, optimizer_selection: str) -> Any:
        if optimizer_selection in optimizers:
            return optimizers[optimizer_selection]
        else:
            self.get_logger().info("The " + optimizer_selection + " is not available")
            raise Exception

    @staticmethod
    def get_logger() -> logging.Logger:
        return logging.getLogger('message_passing_nn')
