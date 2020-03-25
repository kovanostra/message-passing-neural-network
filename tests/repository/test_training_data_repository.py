from unittest import TestCase

from src.repository.training_data_repository import TrainingDataRepository


class TestTrainingDataRepository(TestCase):
    def setUp(self) -> None:
        self.training_data_repository = TrainingDataRepository()

    def test_get_all(self):
        self.fail()
