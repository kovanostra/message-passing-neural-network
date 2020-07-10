from abc import ABCMeta, abstractmethod


class Usecase(metaclass=ABCMeta):
    @abstractmethod
    def start(self) -> None:
        pass
