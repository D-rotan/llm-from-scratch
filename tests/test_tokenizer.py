import pickle
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmfs.configs.schema import TokenizerConfig
from llmfs.tokenizer.tokenizer import Tokenizer


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)
        self.vocab_path = self.directory / "vocab.pkl"
        self.merges_path = self.directory / "merges.pkl"
        with self.vocab_path.open("wb") as file:
            pickle.dump(
                {0: b"a", 1: b"b", 2: b"ab", 3: b"<|end|>", 4: b"\n"},
                file,
            )
        with self.merges_path.open("wb") as file:
            pickle.dump([(b"a", b"b")], file)
        self.tokenizer = Tokenizer(
            TokenizerConfig(
                vocab_filepath=self.vocab_path,
                merges_filepath=self.merges_path,
                special_tokens=["<|end|>"],
            )
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_encode_and_decode_text_with_merge_and_special_token(self):
        ids = self.tokenizer.encode("ab<|end|>")
        self.assertEqual(ids, [2, 3])
        self.assertEqual(self.tokenizer.decode(ids), "ab<|end|>")

    def test_encode_iterable_yields_ids_for_each_line(self):
        ids = list(self.tokenizer.encode_iterable(["ab\n", "b"]))
        self.assertEqual(ids, [2, 4, 1])

    def test_decode_does_not_need_merges_file(self):
        tokenizer = Tokenizer(
            TokenizerConfig(
                vocab_filepath=self.vocab_path,
                merges_filepath=None,
                special_tokens=[],
            )
        )
        self.assertEqual(tokenizer.decode([0, 1]), "ab")


if __name__ == "__main__":
    unittest.main()
