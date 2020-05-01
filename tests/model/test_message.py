from unittest import TestCase

import torch as to

from message_passing_nn.model.message_gru import MessageGRU


class TestMessage(TestCase):
    def setUp(self) -> None:
        device = "cpu"
        self.message = MessageGRU(device)
        self.message.update_gate = to.tensor([1, 2])
        self.message.previous_messages = to.tensor([-2, 2])
        self.message.current_memory = to.tensor([-1, 1])

    def test_compose(self):
        # Given
        message_value_expected = to.tensor([-1, 0]).float()

        # When
        self.message.compose()

        # Then
        self.assertTrue(to.allclose(message_value_expected, self.message.value))
