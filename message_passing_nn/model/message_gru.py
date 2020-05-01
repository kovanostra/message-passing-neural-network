import torch as to

from message_passing_nn.model.message import Message


class MessageGRU(Message):

    def __init__(self, device: str):
        super().__init__()
        self.previous_messages = None
        self.update_gate = None
        self.current_memory = None
        self.device = device

    def compose(self) -> None:
        self.value = to.add(
                            to.mul(
                                   to.sub(to.ones(self.update_gate.shape).to(self.device),
                                          self.update_gate),
                                   self.previous_messages),
                            to.mul(self.update_gate,
                                   self.current_memory))
