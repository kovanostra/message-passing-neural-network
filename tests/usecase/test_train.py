from unittest import TestCase

from src.usecase.train import Train


class TestTrain(TestCase):
    def setUp(self) -> None:
        self.train = Train()

    def test_start(self):
        # Given
        self.train.epochs = 10

        # When
        running_loss = self.train.start()

        # Then
        self.assertTrue(running_loss > 0.0)
