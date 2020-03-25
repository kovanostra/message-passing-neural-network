from src.usecase.train import Train


class MessagePassingNN:
    def __init__(self, train) -> None:
        self.train = train

    def start(self):
        self.train.start()


def create(epochs: int, loss_function: str, optimizer: str) -> MessagePassingNN:
    create_success_file()
    train = Train(epochs, loss_function, optimizer)
    return MessagePassingNN(train)


def create_success_file():
    f = open("SUCCESS", "w")
    f.close()
