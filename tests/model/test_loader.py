from unittest import TestCase

from message_passing_nn.graph import RNNEncoder
from message_passing_nn.model import Loader


class TestLoader(TestCase):
    def test_load_model(self):
        # Given
        loader = Loader("RNN")
        data_dimensions = ([4, 2], [16])
        path_to_model = "tests/test_data/model-checkpoints-test/configuration&id__model&" + \
                        "RNN__epochs&10__loss_function&MSE__optimizer&Adagrad__batch_size&" + \
                        "100__validation_split&0.2__test_split&0.1__time_steps&1__validation_period&" + \
                        "5/Epoch_5_model_state_dictionary.pth"

        # When
        model = loader.load_model(data_dimensions, path_to_model)

        # Then
        self.assertTrue(isinstance(model, RNNEncoder))
