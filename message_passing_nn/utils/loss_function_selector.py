import logging

from torch.nn.modules.module import Module

from message_passing_nn.utils.loss_functions import loss_functions


class LossFunctionSelector:
    def __init__(self) -> None:
        pass

    @staticmethod
    def load_loss_function(loss_function_selection: str) -> Module:
        if loss_function_selection in loss_functions:
            return loss_functions[loss_function_selection]
        else:
            get_logger().info("The " + loss_function_selection + " is not available")
            raise Exception


def get_logger() -> logging.Logger:
    return logging.getLogger('message_passing_nn')
