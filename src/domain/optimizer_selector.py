import logging
from typing import Any

from src.fixtures.optimizers import optimizers


class OptimizerSelector:
    def __init__(self) -> None:
        pass

    @staticmethod
    def load_optimizer(optimizer_selection: str) -> Any:
        if optimizer_selection in optimizers:
            return optimizers[optimizer_selection]
        else:
            get_logger().info("The " + optimizer_selection + " is not available")
            raise Exception


def get_logger() -> logging.Logger:
    return logging.getLogger('message_passing_nn')
