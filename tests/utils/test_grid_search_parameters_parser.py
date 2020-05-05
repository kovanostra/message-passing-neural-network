from unittest import TestCase

from message_passing_nn.utils.grid_search_parameters_parser import GridSearchParametersParser


class TestGridSearchParametersParser(TestCase):
    def test_get_grid_search_dictionary(self):
        # Given
        model = "RNN&GRU"
        epochs = "10&15&5"
        loss_function_selection = "MSE&CrossEntropy"
        optimizer_selection = "SGD&Adam"
        batch_size = "5"
        maximum_number_of_nodes = "-1"
        maximum_number_of_features = "-1"
        validation_split = "0.2"
        test_split = "0.1&0.2&2"
        time_steps = "10"
        validation_period = "5"
        grid_search_dictionary_expected = {
            "model": ["RNN", "GRU"],
            "epochs": [10, 11, 12, 13, 15],
            "loss_function": ["MSE", "CrossEntropy"],
            "optimizer": ["SGD", "Adam"],
            "batch_size": [5],
            "maximum_number_of_nodes": [-1],
            "maximum_number_of_features": [-1],
            "validation_split": [0.2],
            "test_split": [0.1, 0.2],
            "time_steps": [10],
            "validation_period": [5],
        }

        # When
        grid_search_dictionary = GridSearchParametersParser().get_grid_search_dictionary(
            model,
            epochs,
            loss_function_selection,
            optimizer_selection,
            batch_size,
            maximum_number_of_nodes,
            maximum_number_of_features,
            validation_split,
            test_split,
            time_steps,
            validation_period)

        # Then
        self.assertEqual(grid_search_dictionary_expected, grid_search_dictionary)
