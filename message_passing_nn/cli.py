import logging

import click
import sys

from message_passing_nn.create_message_passing_nn import create


@click.group("message-passing-nn")
@click.option('--debug', default=False, help='Set the logs to debug level', show_default=True, is_flag=True)
def main(debug):
    log_level = logging.DEBUG if debug else logging.INFO
    setup_logging(log_level)


@click.command('grid-search', help='Starts the grid search')
@click.argument('dataset_name', envvar='DATASET_NAME', type=str)
@click.argument('data_directory', envvar='DATA_DIRECTORY', type=str)
@click.argument('model_directory', envvar='MODEL_DIRECTORY', type=str)
@click.argument('results_directory', envvar='RESULTS_DIRECTORY', type=str)
@click.argument('device', envvar='DEVICE', type=str)
@click.argument('epochs', envvar='EPOCHS', type=str)
@click.argument('loss_function', envvar='LOSS_FUNCTION', type=str)
@click.argument('optimizer', envvar='OPTIMIZER', type=str)
@click.argument('batch_size', envvar='BATCH_SIZE', type=str)
@click.argument('validation_split', envvar='VALIDATION_SPLIT', type=str)
@click.argument('test_split', envvar='TEST_SPLIT', type=str)
@click.argument('time_steps', envvar='TIME_STEPS', type=str)
@click.argument('validation_period', envvar='VALIDATION_PERIOD', type=str)
def start_training(dataset_name: str,
                   data_directory: str,
                   model_directory: str,
                   results_directory: str,
                   device: str,
                   epochs: str,
                   loss_function: str,
                   optimizer: str,
                   batch_size: str,
                   validation_split: str,
                   test_split: str,
                   time_steps: str,
                   validation_period: str) -> None:
    get_logger().info("Starting training")
    message_passing_nn = create(dataset_name,
                                data_directory,
                                model_directory,
                                results_directory,
                                device,
                                epochs,
                                loss_function,
                                optimizer,
                                batch_size,
                                validation_split,
                                test_split,
                                time_steps,
                                validation_period)
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
