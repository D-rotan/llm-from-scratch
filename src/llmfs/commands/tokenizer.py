import argparse
from pathlib import Path

from ..configs.schema import DecodeRequest, EncodeRequest, TokenizerConfig, TrainRequest
from ..tokenizer.tokenizer import Tokenizer

import os
import time

def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("tokenize", help="编码或解码文本")
    operations = parser.add_subparsers(dest="operation", required=True)

    train_mode = operations.add_parser("train_mode",help="用于大段文本编码为 Token ID，训练模型")
    add_files_arguments(train_mode,require_merges=True)
    train_mode.add_argument("--text_path",type=Path,required=True,help="需要用于训练的编码文本路径")
    train_mode.add_argument("--save_path",type=Path,help="用于保存训练的编码文本路径",default=Path("data/tokenizer/"))
    train_mode.set_defaults(handler=run_train_mode)


    encode = operations.add_parser("encode", help="文本编码为 token id")
    add_files_arguments(encode, require_merges=True)
    encode.add_argument("--text", required=True, help="需要编码的文本")
    encode.set_defaults(handler=run_encode)

    decode = operations.add_parser("decode", help="token id 解码为文本")
    add_files_arguments(decode, require_merges=False)
    decode.add_argument("--ids", type=int, nargs="+", required=True, help="token id 列表")
    decode.set_defaults(handler=run_decode)


def add_files_arguments(parser: argparse.ArgumentParser, *, require_merges: bool) -> None:
    parser.add_argument("--vocab-file", type=Path, required=True, help="词表路径")
    parser.add_argument(
        "--merges-file",
        type=Path,
        required=require_merges,
        help="merges 路径；encode 必须提供",
    )
    parser.add_argument(
        "--special-token",
        dest="special_tokens",
        action="append",
        default=["<|endoftext|>"],
        help="特殊 Token；可以重复传入",
    )


def make_config(args: argparse.Namespace) -> TokenizerConfig:
    return TokenizerConfig(
        vocab_filepath=args.vocab_file,
        merges_filepath=args.merges_file,
        special_tokens=args.special_tokens,
    )

def run_train_mode(args: argparse.Namespace) -> None:
    import numpy as np

    config = make_config(args)
    request = TrainRequest(text_path=args.text_path,save_path=args.save_path)
    tokenizer = Tokenizer(config=config)
    print("进入训练编码模式（过程中请不要终止程序，否则可能会导致语料编码出错）")
    request.save_path.parent.mkdir(parents=True, exist_ok=True)
    token_list = []
    times = 0
    count = 0
    token_count = 0
    start = time.perf_counter()
    save_path = request.save_path / request.text_path.stem
    save_path.mkdir(exist_ok=True,parents=True)
    with open(request.text_path,"r",encoding="utf-8") as file:
        for token in tokenizer.encode_iterable(file):
            times += 1
            token_list.append(token)
            if times % 20_000_000 == 0:
                count += 1
                token_count += len(token_list)
                print(f"已编码 Token 数：{token_count}")
                np.save(Path(f"{save_path / request.text_path.stem}{count}.npy"),token_list)
                token_list = []
        if token_list:
            count += 1
            token_count += len(token_list)
            np.save(Path(f"{save_path / request.text_path.stem}{count}.npy"),token_list)
            token_list = []
    end = time.perf_counter()
    print("一共编码 Token 数：",token_count)
    print(f"总共耗时：{(end - start) * 1000:.2f} ms")
    start = time.perf_counter()
    print("现在进行总和文件")
    output_path = (save_path / request.text_path.stem).with_suffix(".npy")
    data = np.lib.format.open_memmap(output_path,dtype=np.int64,shape=(token_count,),mode="w+")
    print("开始")
    offset = 0
    for i in range(count):
        shard_path = Path(f"{save_path / request.text_path.stem}{i+1}.npy")
        shard = np.load(shard_path,mmap_mode="r")
        data[offset:offset + len(shard)] = shard
        offset += len(shard)

        shard._mmap.close()
        del shard

        os.remove(shard_path)
    data.flush()
    data._mmap.close()
    del data
    print("保存成功")
    end = time.perf_counter()
    print(f"总共耗时：{(end - start):.4f} s")

def run_encode(args: argparse.Namespace) -> None:
    config = make_config(args)
    request = EncodeRequest(text=args.text)
    print("进入编码模式，以下是编码结果")
    print(Tokenizer(config).encode(request.text))
    print("编码结束")


def run_decode(args: argparse.Namespace) -> None:
    config = make_config(args)
    request = DecodeRequest(ids=args.ids)
    print("进入解码模式，以下是解码结果")
    print(Tokenizer(config).decode(request.ids))
    print("解码结束")
