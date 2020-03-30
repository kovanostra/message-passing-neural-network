from unittest import TestCase

import torch as to

from src.domain.fully_connected_layer import FullyConnectedLayer


class TestFullyConnectedLayer(TestCase):
    def test_forward(self):
        # Given
        inputs = to.tensor([1.0, 0.0, 2.0, 1.0])
        input_size = 4
        output_size = 2
        fully_connected_layer = FullyConnectedLayer(input_size, output_size)
        fully_connected_layer.linear.weight = to.nn.Parameter(to.ones(output_size, input_size), requires_grad=False)
        fully_connected_layer.linear.bias = to.nn.Parameter(2 * to.ones(output_size), requires_grad=False)
        outputs_expected = to.tensor([0.9975274, 0.9975274])

        # When
        outputs = fully_connected_layer.forward(inputs)

        # Then
        self.assertTrue(to.allclose(outputs_expected, outputs))
