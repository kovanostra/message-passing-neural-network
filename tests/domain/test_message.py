from unittest import TestCase

import numpy as np

from src.domain.message_gru import Message


class TestMessage(TestCase):
    def setUp(self) -> None:
        self.message = Message()
        self.message.update_gate = np.array([1, 2])
        self.message.previous_messages = np.array([-2, 2])
        self.message.current_memory = np.array([-1, 1])

    def test_compose(self):
        # Given
        message_value_expected = np.array([-1, 0])

        # When
        self.message.compose()

        # Then
        self.assertTrue(np.array_equal(message_value_expected, self.message.value))
