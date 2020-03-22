from src.domain.interface.message import Message


class MessageGRU(Message):

    def __init__(self):
        super().__init__()
        self.previous_messages = None
        self.update_gate = None
        self.current_memory = None

    def compose(self) -> None:
        self.value = (1 - self.update_gate) * self.previous_messages + self.update_gate * self.current_memory
