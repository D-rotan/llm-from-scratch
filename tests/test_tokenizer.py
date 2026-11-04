import pickle
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmfs.configs.schema import TokenizerConfig
from llmfs.commands.tokenizer import run_encode
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
        ids = list(self.tokenizer.encode("ab<|end|>"))
        self.assertEqual(ids, [2, 3])
        self.assertEqual(self.tokenizer.decode(ids), "ab<|end|>")

    def test_encode_cache_handles_first_and_repeated_occurrences(self):
        self.assertEqual(list(self.tokenizer.encode("ab")), [2])
        self.assertEqual(list(self.tokenizer.encode("ab")), [2])
        self.assertEqual(self.tokenizer.token_cache["ab"], (2,))

    def test_encode_uses_highest_priority_merge_first(self):
        vocab = {
            0: b"a",
            1: b"b",
            2: b"c",
            3: b"ab",
            4: b"bc",
        }
        vocab_path = self.directory / "priority-vocab.pkl"
        merges_path = self.directory / "priority-merges.pkl"
        with vocab_path.open("wb") as file:
            pickle.dump(vocab, file)
        with merges_path.open("wb") as file:
            # (b, c) has the higher priority because it appears first.
            pickle.dump([(b"b", b"c"), (b"a", b"b")], file)

        tokenizer = Tokenizer(
            TokenizerConfig(vocab_path, merges_path, [])
        )
        self.assertEqual(list(tokenizer.encode("abc")), [0, 4])

    def test_encode_matches_longest_special_token(self):
        vocab = {
            0: b"a",
            1: b"<|end|>",
            2: b"<|end|><|end|>",
        }
        vocab_path = self.directory / "special-vocab.pkl"
        merges_path = self.directory / "special-merges.pkl"
        with vocab_path.open("wb") as file:
            pickle.dump(vocab, file)
        with merges_path.open("wb") as file:
            pickle.dump([], file)

        tokenizer = Tokenizer(
            TokenizerConfig(
                vocab_path,
                merges_path,
                ["<|end|>", "<|end|><|end|>"],
            )
        )
        self.assertEqual(
            list(tokenizer.encode("a<|end|><|end|>")),
            [0, 2],
        )

    def test_utf8_text_round_trips(self):
        vocab = {index: bytes([index]) for index in range(256)}
        vocab_path = self.directory / "utf8-vocab.pkl"
        merges_path = self.directory / "utf8-merges.pkl"
        with vocab_path.open("wb") as file:
            pickle.dump(vocab, file)
        with merges_path.open("wb") as file:
            pickle.dump([], file)

        tokenizer = Tokenizer(TokenizerConfig(vocab_path, merges_path, []))
        text = "你好，世界！ 😀"
        ids = list(tokenizer.encode(text))
        self.assertEqual(tokenizer.decode(ids), text)

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

    def test_cli_encode_prints_token_ids(self):
        args = type("Args", (), {
            "vocab_file": self.vocab_path,
            "merges_file": self.merges_path,
            "special_tokens": ["<|end|>"],
            "text": "ab<|end|>",
        })()
        output = StringIO()
        with redirect_stdout(output):
            run_encode(args)

        self.assertIn("[2, 3]", output.getvalue())


if __name__ == "__main__":
    unittest.main()
