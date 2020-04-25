import torch as to

from src.domain.interface.message import Message


class MessageGRU(Message):

    def __init__(self):
        super().__init__()
        self.previous_messages = None
        self.update_gate = None
        self.current_memory = None

    def compose(self) -> None:
        self.value = to.add(
                            to.mul(
                                   to.sub(to.ones(self.update_gate.shape),
                                          self.update_gate),
                                   self.previous_messages),
                            to.mul(self.update_gate,
                                   self.current_memory))
