import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmfs.cli import build_parser
from llmfs.commands import generate, model_train, train_bpe
from llmfs.commands.presets import MODEL_PRESETS
from llmfs.configs.schema import BPETrainerConfig, GenerationConfig, ModelTrainerConfig


class CommandLineTests(unittest.TestCase):
    def test_cli_registers_all_modes(self):
        parser = build_parser()
        command_action = next(action for action in parser._actions if action.dest == "command")
        self.assertEqual(
            set(command_action.choices),
            {"bpe-train", "model-train", "tokenize", "generate"},
        )

    def test_bpe_arguments_are_passed_as_config(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "bpe-train",
                "--input-path", "corpus.txt",
                "--vocab-size", "300",
                "--special-token", "<|end|>",
                "--num-processes", "2",
                "--outputs", "tokenizer",
            ]
        )
        with patch.object(train_bpe, "BPE_Train") as train:
            args.handler(args)

        config = train.call_args.args[0]
        self.assertIsInstance(config, BPETrainerConfig)
        self.assertEqual(config.input_path, Path("corpus.txt"))
        self.assertEqual(config.vocab_size, 300)
        # --special-token 会追加到默认的结束标记之后。
        self.assertEqual(config.special_tokens, ["<|endoftext|>", "<|end|>"])
        self.assertEqual(config.num_processes, 2)
        self.assertEqual(config.outputs, Path("tokenizer"))

    def test_model_train_arguments_are_passed_as_config(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "model-train", "--src", "tokens.npy", "--max-iters", "8",
                "--vocab-size", "300", "--context-length", "16", "--num-layers", "2",
                "--d-model", "32", "--d-ff", "64", "--num-heads", "4",
                "--rope-theta", "10000", "--batch-size", "2", "--lr-max", "0.001",
                "--lr-min", "0.0001", "--warmup-iters", "1",
                "--cosine-cycle-iters", "8", "--max-l2-norm", "1.0",
                "--checkpoint-iters", "2", "--out", "model.pt", "--betas", "0.8", "0.95",
            ]
        )
        training_module = types.ModuleType("llmfs.training.model_trainer")
        from unittest.mock import Mock
        training_module.Training_loop = Mock()
        with patch.dict(sys.modules, {"llmfs.training.model_trainer": training_module}):
            args.handler(args)

        config = training_module.Training_loop.call_args.args[0]
        self.assertIsInstance(config, ModelTrainerConfig)
        self.assertEqual(config.src, Path("tokens.npy"))
        self.assertEqual(config.betas, (0.8, 0.95))
        self.assertEqual(config.out, Path("model.pt"))

    def test_generate_arguments_are_passed_as_config(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "generate", "--params-path", "model.pt", "--vocab-file", "vocab.pkl",
                "--merges-file", "merges.pkl", "--prompt", "hello", "--top-p", "0.9",
                "--vocab-size", "300", "--context-length", "16", "--num-layers", "2",
                "--d-model", "32", "--d-ff", "64", "--num-heads", "4",
                "--rope-theta", "10000",
            ]
        )
        generation_module = types.ModuleType("llmfs.generation.generator")
        from unittest.mock import Mock
        generation_module.generate = Mock(return_value="hello")
        with patch.dict(sys.modules, {"llmfs.generation.generator": generation_module}):
            args.handler(args)

        config = generation_module.generate.call_args.args[0]
        self.assertIsInstance(config, GenerationConfig)
        self.assertEqual(config.params_path, Path("model.pt"))
        self.assertEqual(config.prompt, "hello")
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.tokenizer.vocab_filepath, Path("vocab.pkl"))

    def test_model_preset_supplies_and_allows_overriding_model_shape(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "model-train", "--preset", "small", "--src", "tokens.npy",
                "--max-iters", "8", "--vocab-size", "300", "--d-model", "256",
                "--batch-size", "2", "--lr-max", "0.001", "--lr-min", "0.0001",
                "--warmup-iters", "1", "--cosine-cycle-iters", "8",
                "--max-l2-norm", "1.0", "--checkpoint-iters", "2", "--out", "model.pt",
            ]
        )
        from unittest.mock import Mock
        training_module = types.ModuleType("llmfs.training.model_trainer")
        training_module.Training_loop = Mock()
        with patch.dict(sys.modules, {"llmfs.training.model_trainer": training_module}):
            args.handler(args)

        config = training_module.Training_loop.call_args.args[0]
        self.assertEqual(config.context_length, MODEL_PRESETS["small"]["context_length"])
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.num_layers, MODEL_PRESETS["small"]["num_layers"])

    def test_model_train_preset_needs_no_other_arguments(self):
        parser = build_parser()
        args = parser.parse_args(["model-train", "--preset", "small"])
        from unittest.mock import Mock
        training_module = types.ModuleType("llmfs.training.model_trainer")
        training_module.Training_loop = Mock()
        with patch.dict(sys.modules, {"llmfs.training.model_trainer": training_module}):
            args.handler(args)

        config = training_module.Training_loop.call_args.args[0]
        self.assertIsInstance(config, ModelTrainerConfig)
        self.assertEqual(config.src, Path("data/tokenizer.npy"))
        self.assertEqual(config.out, Path("data/model_params/small.pt"))

    def test_generate_preset_uses_the_same_model_shape(self):
        parser = build_parser()
        args = parser.parse_args(
            ["generate", "--preset", "mid"]
        )
        from unittest.mock import Mock
        generation_module = types.ModuleType("llmfs.generation.generator")
        generation_module.generate = Mock(return_value="hello")
        with patch.dict(sys.modules, {"llmfs.generation.generator": generation_module}):
            args.handler(args)

        config = generation_module.generate.call_args.args[0]
        for field in (
            "context_length", "num_layers", "d_model", "d_ff", "num_heads",
            "rope_theta", "device",
        ):
            value = MODEL_PRESETS["mid"][field]
            self.assertEqual(getattr(config, field), value)
        self.assertEqual(config.vocab_size, MODEL_PRESETS["mid"]["vocab_size"])
        self.assertEqual(config.prompt, "你好")
        self.assertEqual(config.params_path, Path("data/model_params/mid.pt"))


if __name__ == "__main__":
    unittest.main()
