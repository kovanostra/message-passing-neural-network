import os.path
from os import path
from unittest import TestCase

from message_passing_nn.infrastructure.file_system_repository import FileSystemRepository
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES


class TestTrainingDataRepository(TestCase):
    def setUp(self) -> None:
        self.dataset = 'repo-test-data'
        self.tests_data_directory = 'tests/test_data/'
        self.file_system_repository = FileSystemRepository(self.tests_data_directory, self.dataset)

    def test_save(self):
        # Given
        features = BASE_GRAPH_NODE_FEATURES
        adjacency_matrix = BASE_GRAPH
        labels = BASE_GRAPH.view(-1)

        filenames_to_save = ['code_features.pickle', 'code_adjacency-matrix.pickle', 'code_labels.pickle']
        filenames_expected = [self.tests_data_directory + self.dataset + '/code_features.pickle',
                              self.tests_data_directory + self.dataset + '/code_adjacency-matrix.pickle',
                              self.tests_data_directory + self.dataset + '/code_labels.pickle']

        # When
        self.file_system_repository.save(filenames_to_save[0], features)
        self.file_system_repository.save(filenames_to_save[1], adjacency_matrix)
        self.file_system_repository.save(filenames_to_save[2], labels)

        # Then
        path.exists(filenames_expected[0])
        path.exists(filenames_expected[1])
        path.exists(filenames_expected[2])
        os.remove(filenames_expected[0])
        os.remove(filenames_expected[1])
        os.remove(filenames_expected[2])
