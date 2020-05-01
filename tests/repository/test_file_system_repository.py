import os.path
from os import path
from unittest import TestCase

import torch as to

from message_passing_nn.repository.file_system_repository import FileSystemRepository
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES


class TestTrainingDataRepository(TestCase):
    def setUp(self) -> None:
        self.dataset = 'repo-test-data'
        self.tests_data_directory = 'tests/test_data/'
        self.file_system_repository = FileSystemRepository(self.tests_data_directory, self.dataset)

    def test_save(self):
        # Given
        features = BASE_GRAPH_NODE_FEATURES
        labels = BASE_GRAPH

        filenames_to_save = ['code_features.pickle', 'code_labels.pickle']
        filenames_expected = [self.tests_data_directory + self.dataset + '/code_features.pickle',
                              self.tests_data_directory + self.dataset + '/code_labels.pickle']

        # When
        self.file_system_repository.save(filenames_to_save[0], features)
        self.file_system_repository.save(filenames_to_save[1], labels)

        # Then
        path.exists(filenames_expected[0])
        path.exists(filenames_expected[1])
        os.remove(filenames_expected[0])
        os.remove(filenames_expected[1])

    def test_get_all_features_and_labels_from_separate_files(self):
        # Given
        features_expected = BASE_GRAPH_NODE_FEATURES
        labels_expected = BASE_GRAPH
        filenames_to_save = ['code_features.pickle', 'code_labels.pickle']
        self.file_system_repository.save(filenames_to_save[0], features_expected)
        self.file_system_repository.save(filenames_to_save[1], labels_expected)
        all_data_expected = [(features_expected, labels_expected)]

        # When
        all_data = self.file_system_repository.get_all_features_and_labels_from_separate_files()

        # Then
        self.assertTrue(to.allclose(all_data_expected[0][0], all_data[0][0]))
        self.assertTrue(to.allclose(all_data_expected[0][1], all_data[0][1]))
