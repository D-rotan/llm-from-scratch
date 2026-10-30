import argparse

from .commands import generate, model_train, tokenizer, train_bpe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="llm-from-scratch 命令行工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_bpe.add_parser(subparsers)
    model_train.add_parser(subparsers)
    tokenizer.add_parser(subparsers)
    generate.add_parser(subparsers)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
