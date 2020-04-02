from typing import Any

from src.fixtures.optimizers import optimizers


class OptimizerSelector:
    def __init__(self, optimizer_selection: str) -> None:
        self.optimizer = self.optimizer_loader(optimizer_selection)

    @staticmethod
    def optimizer_loader(optimizer_selection: str) -> Any:
        return optimizers[optimizer_selection]
