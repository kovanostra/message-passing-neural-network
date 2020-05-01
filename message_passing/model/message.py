from abc import ABCMeta, abstractmethod


class Message(metaclass=ABCMeta):
    def __init__(self):
        self.value = None

    @abstractmethod
    def compose(self) -> None:
        pass
