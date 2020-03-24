from unittest import TestCase

from src.usecase.train import Train


class TestTrain(TestCase):
    def setUp(self) -> None:
        self.train = None

    def test_start(self):
        # Given
        self.train = Train(epochs=10, loss_function='MSE', optimizer='adam')

        # When
        running_loss = self.train.start()

        # Then
        self.assertTrue(running_loss > 0.0)
