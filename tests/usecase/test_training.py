import os
from unittest import TestCase

import torch as to
from torch import nn

from src.domain.graph_encoder import GraphEncoder
from src.domain.model_trainer import ModelTrainer
from src.repository.training_data_repository import TrainingDataRepository
from src.usecase.training import Training
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES


class TestTraining(TestCase):
    def test_start_for_multiple_batches_of_the_same_size(self):
        # Given
        batch_size = 3
        dataset_size = 6
        validation_split = 0.2
        test_split = 0.1
        loss_function = nn.MSELoss()
        optimizer = to.optim.SGD
        dataset = 'training-test-data'
        tests_data_path = 'tests/data/'
        repository = TrainingDataRepository(tests_data_path, dataset)
        model_trainer = ModelTrainer(GraphEncoder, loss_function, optimizer)
        training = Training(repository, model_trainer, epochs=10)

        features = BASE_GRAPH_NODE_FEATURES
        labels = BASE_GRAPH
        features_filenames = [str(i) + '_training_features' + '.pickle' for i in range(dataset_size)]
        labels_filenames = [str(i) + '_training_labels' '.pickle' for i in range(dataset_size)]
        for i in range(dataset_size):
            repository.save(features_filenames[i], features)
            repository.save(labels_filenames[i], labels)

        # When
        training_loss, validation_loss, test_loss = training.start(batch_size, validation_split, test_split)

        # Then
        self.assertTrue(training_loss > 0.0)
        self.assertTrue(validation_loss > 0.0)
        self.assertTrue(test_loss > 0.0)
        for i in range(dataset_size):
            os.remove(tests_data_path + dataset + "/" + features_filenames[i])
            os.remove(tests_data_path + dataset + "/" + labels_filenames[i])

    def test_start_for_multiple_batches_of_differing_size(self):
        # Given
        batch_size = 3
        dataset_size = 5
        validation_split = 0.2
        test_split = 0.1
        loss_function = nn.MSELoss()
        optimizer = to.optim.SGD
        dataset = 'training-test-data'
        tests_data_path = 'tests/data/'
        repository = TrainingDataRepository(tests_data_path, dataset)
        model_trainer = ModelTrainer(GraphEncoder, loss_function, optimizer)
        training = Training(repository, model_trainer, epochs=10)

        features = BASE_GRAPH_NODE_FEATURES
        labels = BASE_GRAPH
        features_filenames = [str(i) + '_training_features' + '.pickle' for i in range(dataset_size)]
        labels_filenames = [str(i) + '_training_labels' '.pickle' for i in range(dataset_size)]
        for i in range(dataset_size):
            repository.save(features_filenames[i], features)
            repository.save(labels_filenames[i], labels)

        # When
        training_loss, validation_loss, test_loss = training.start(batch_size, validation_split, test_split)

        # Then
        self.assertTrue(training_loss > 0.0)
        self.assertTrue(validation_loss > 0.0)
        self.assertTrue(test_loss > 0.0)
        for i in range(dataset_size):
            os.remove(tests_data_path + dataset + "/" + features_filenames[i])
            os.remove(tests_data_path + dataset + "/" + labels_filenames[i])
