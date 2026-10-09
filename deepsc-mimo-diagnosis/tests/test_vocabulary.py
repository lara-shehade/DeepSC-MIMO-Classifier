from pathlib import Path
import tempfile
import unittest

from main import (
    EXPECTED_SHARED_VOCAB_SIZE,
    EXPECTED_SPECIAL_TOKEN_IDS,
    resolve_vocab,
    validate_vocabulary,
)


class VocabularySafetyTests(unittest.TestCase):
    def test_resolve_shared_vocab_from_shared_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "shared_vocab.json"
            expected.write_text("{}", encoding="utf-8")
            self.assertEqual(resolve_vocab(data_root=root), expected.resolve())

    def test_resolve_shared_vocab_from_domain_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            domain = root / "europarl"
            domain.mkdir()
            expected = root / "shared_vocab.json"
            expected.write_text("{}", encoding="utf-8")
            self.assertEqual(resolve_vocab(data_root=domain), expected.resolve())

    def test_strict_validation_rejects_old_or_small_vocab(self):
        vocabulary = {"token_to_idx": dict(EXPECTED_SPECIAL_TOKEN_IDS)}
        with self.assertRaisesRegex(ValueError, "90,746"):
            validate_vocabulary(vocabulary)
        self.assertEqual(
            validate_vocabulary(vocabulary, allow_nonshared=True),
            vocabulary["token_to_idx"],
        )

    def test_strict_validation_accepts_exact_shared_profile(self):
        token_to_idx = dict(EXPECTED_SPECIAL_TOKEN_IDS)
        for index in range(len(token_to_idx), EXPECTED_SHARED_VOCAB_SIZE):
            token_to_idx["token_{}".format(index)] = index
        self.assertIs(
            validate_vocabulary({"token_to_idx": token_to_idx}), token_to_idx
        )


if __name__ == "__main__":
    unittest.main()
