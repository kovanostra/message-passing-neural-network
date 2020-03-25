import logging
import sys

import click

from src.message_passing_nn import create


@click.group("message-passing-nn")
@click.option('--debug', default=False, help='Set the logs to debug level', show_default=True, is_flag=True)
def main(debug):
    log_level = logging.DEBUG if debug else logging.INFO
    setup_logging(log_level)


@click.command("start-training", help='Starts the training')
@click.option('--epochs', default=10, help='Set the number of epochs', show_default=True, type=int)
@click.option('--loss_function', default='MSE', help='Set the loss function', show_default=True, type=str)
@click.option('--optimizer', default='adam', help='Set the optimizer', show_default=True, type=str)
def start_training(epochs: int, loss_function: str, optimizer: str) -> None:
    get_logger().info("Starting training")
    message_passing_nn = create(epochs, loss_function, optimizer)
    message_passing_nn.start()


def setup_logging(log_level):
    get_logger().setLevel(log_level)

    logOutputFormatter = logging.Formatter(
        '%(asctime)s %(levelname)s - %(message)s [%(filename)s:%(lineno)s] [%(relativeCreated)d]')

    stdoutStreamHandler = logging.StreamHandler(sys.stdout)
    stdoutStreamHandler.setLevel(log_level)
    stdoutStreamHandler.setFormatter(logOutputFormatter)

    get_logger().addHandler(stdoutStreamHandler)

    stderrStreamHandler = logging.StreamHandler(sys.stdout)
    stderrStreamHandler.setLevel(logging.WARNING)
    stderrStreamHandler.setFormatter(logOutputFormatter)

    get_logger().addHandler(stderrStreamHandler)


def get_logger() -> logging.Logger:
    return logging.getLogger('message_passing_nn')


main.add_command(start_training)

if __name__ == '__main__':
    main()