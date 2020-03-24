import logging
import sys

import click

from src.message_passing_nn import create


@click.group('message_passing_nn')
@click.option('--debug', default=False, help='Set the logs to debug level', is_flag=True)
def main(debug):
    log_level = logging.DEBUG if debug else logging.INFO
    setup_logging(log_level)


@click.command('start_training', help='Starts the training')
@click.argument('epochs', default=10, envvar='epochs', type=int)
@click.argument('loss_function', default='MSE', envvar='loss_function', type=str)
@click.argument('optimizer', default='adam', envvar='optimizer', type=str)
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
