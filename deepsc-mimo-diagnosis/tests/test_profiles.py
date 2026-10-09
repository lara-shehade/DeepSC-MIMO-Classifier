import random
import unittest

from profiles import DEFAULT_OP_WEIGHTS, corrupt_sequence


class LiteralProfileTests(unittest.TestCase):
    def test_previous_literal_mix_and_special_token_protection(self):
        self.assertEqual(DEFAULT_OP_WEIGHTS, {
            "deletion": 0.35, "typo": 0.30,
            "insertion_repeat": 0.20, "grammar_replacement": 0.15,
        })
        vocabulary = {
            "<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3,
            "this": 4, "these": 5, "example": 6, "sentence": 7,
            "is": 8, "are": 9, "long": 10, "enough": 11,
        }
        reverse = {value: key for key, value in vocabulary.items()}
        clean = [1, 4, 6, 7, 8, 10, 11, 2]
        corrupted, metadata = corrupt_sequence(
            clean, reverse, vocabulary, random.Random(123), severity=0.20
        )
        self.assertEqual(corrupted[0], 1)
        self.assertEqual(corrupted[-1], 2)
        self.assertNotIn(0, corrupted)
        self.assertEqual(metadata["severity_requested"], 0.20)
        self.assertGreaterEqual(metadata["events_applied"], 1)


if __name__ == "__main__":
    unittest.main()
