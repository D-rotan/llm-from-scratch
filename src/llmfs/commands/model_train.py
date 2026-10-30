import argparse
from pathlib import Path

from ..configs.schema import ModelTrainerConfig
from .presets import MODEL_PRESETS, resolve_settings

import time

def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "model-train",
        help="训练语言模型",
    )
    add_model_arguments(parser)
    parser.set_defaults(handler=run)


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=MODEL_PRESETS, help="模型规格预设，仍需指定Token ID 数据文件位置")
    parser.add_argument("--src", type=Path, help="Token ID 数据文件")
    parser.add_argument("--max-iters", type=int,help="最大训练次数")
    parser.add_argument("--vocab-size", type=int,help="词表大小")
    parser.add_argument("--context-length", type=int,help="上下文长度")
    parser.add_argument("--num-layers", type=int,help="Transformer模块层数")
    parser.add_argument("--d-model", type=int,help="模型隐藏维度")
    parser.add_argument("--d-ff", type=int,help="FFN 中间层维度")
    parser.add_argument("--num-heads", type=int,help="注意力头数")
    parser.add_argument("--rope-theta", type=float,help="RoPE 的基础频率参数")
    parser.add_argument("--batch-size", type=int,help="每个训练批次的样本数")
    parser.add_argument("--device",help="设备")
    parser.add_argument("--lr-max", type=float,help="最大学习率")
    parser.add_argument("--lr-min", type=float,help="最小学习率")
    parser.add_argument("--warmup-iters", type=int,help="	学习率预热步数")
    parser.add_argument("--cosine-cycle-iters", type=int,help="余弦退火周期步数")
    parser.add_argument("--max-l2-norm", type=float,help="梯度裁剪的最大 L2 范数")
    parser.add_argument("--checkpoint-iters", type=int,help="检查点保存间隔（训练步数）")
    parser.add_argument("--out", type=Path,default=None,help="参数存放位置")
    parser.add_argument("--lr", type=float,help="学习率，开启学习率调整时可不用输入")
    parser.add_argument(
        "--on-gradient-clipping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否开启梯度裁切"
    )
    parser.add_argument(
        "--on-learning-rate-schedule",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否开启学习率调整"
    )
    parser.add_argument(
    "--betas",
    type=float,
    nargs=2,
    default=None,
    metavar=("B1", "B2"),
    help="AdamW 的一阶和二阶动量系数"
    )

    parser.add_argument("--weight-decay", type=float, help="AdamW 权重衰减系数")

    parser.add_argument("--eps-rms1", dest="eps_RMS1", type=float, help="RMSNorm 第一个数值稳定参数")
    parser.add_argument("--eps-rms2", dest="eps_RMS2", type=float, help="RMSNorm 第二个数值稳定参数")
    parser.add_argument("--eps-rms3", dest="eps_RMS3", type=float, help="RMSNorm 第三个数值稳定参数")
    parser.add_argument("--eps-adamw", dest="eps_AdamW", type=float, help="AdamW 数值稳定参数")
    parser.add_argument("--eps-gradient", type=float, help="梯度裁剪数值稳定参数")


def run(args: argparse.Namespace) -> None:
    from ..training.model_trainer import Training_loop
    model_settings = resolve_settings(
        args.preset,
        args,
        MODEL_PRESETS,
        required_fields=(
            "src","max_iters", "vocab_size", "context_length", "num_layers",
            "d_model", "d_ff", "num_heads", "rope_theta", "batch_size",
            "lr_max", "lr_min", "warmup_iters", "cosine_cycle_iters", "max_l2_norm",
            "checkpoint_iters", "out",
        ),
    )
    model_settings.setdefault("device", "cpu")
    if isinstance(model_settings.get("betas"), list):
        model_settings["betas"] = tuple(model_settings["betas"])
    try:
        config = ModelTrainerConfig(
            **model_settings,
        )
    except TypeError as error:
        raise ValueError(f"训练配置不完整：{error}") from error
    print("正在训练模型")
    start = time.perf_counter()
    Training_loop(config)
    end = time.perf_counter()
    print(f"模型训练结束，总共耗时：{(end - start):.4f} s")

