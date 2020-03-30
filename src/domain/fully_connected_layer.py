from typing import Any

import torch as to


class FullyConnectedLayer(to.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fully_connected_layer = to.nn.Linear(self.input_size, self.output_size)
        self.sigmoid = to.nn.Sigmoid()

    def forward(self, input_data: Any) -> Any:
        output = self.fully_connected_layer(input_data)
        output = self.sigmoid(output)
        return output
