import argparse
from pathlib import Path

from ..configs.schema import BPETrainerConfig
from ..tokenizer.bpe_trainer import BPE_Train

import time

def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("bpe-train", help="从文本训练 BPE Tokenizer")
    parser.add_argument("--input-path", type=Path, required=True, help="语料文件路径")
    parser.add_argument("--vocab-size", type=int, required=True, help="词表大小")
    parser.add_argument(
        "--special-token",
        dest="special_tokens",
        action="append",
        default=["<|endoftext|>"],
        help="特殊 Token；可以重复传入",
    )
    parser.add_argument("--num-processes", type=int, default=None, help="多进程数量")
    parser.add_argument("--outputs", type=Path, help="输出目录",default="data/bpe_train")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> None:
    config = BPETrainerConfig(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=args.num_processes,
        outputs=args.outputs,
    )
    print("正在训练BPE_Tokenzier")
    start = time.perf_counter()
    BPE_Train(config)
    end = time.perf_counter()
    print(f"训练结束，总共耗时：{(end - start):.4f} s")
