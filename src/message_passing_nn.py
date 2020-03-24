from src.usecase.train import Train


class MessagePassingNN:
    def __init__(self, train) -> None:
        self.train = train

    def start(self):
        self.train.start()


def create(epochs: int, loss_function: str, optimizer: str) -> MessagePassingNN:
    train = Train(epochs, loss_function, optimizer)
    return MessagePassingNN(train)
