from typing import Any

from src.fixtures.loss_functions import loss_functions


class LossFunctionSelector:
    def __init__(self, loss_function_selection: str) -> None:
        self.loss_function = self.loss_function_loader(loss_function_selection)

    @staticmethod
    def loss_function_loader(loss_function_selection: str) -> Any:
        return loss_functions[loss_function_selection]
