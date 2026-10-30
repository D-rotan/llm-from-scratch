import argparse
from pathlib import Path

from ..configs.schema import GenerationConfig, TokenizerConfig
from .presets import GENERATION_PRESETS, resolve_settings


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("generate", help="根据提示词生成文本")
    parser.add_argument("--preset", choices=GENERATION_PRESETS, help="生成预设")
    parser.add_argument("--params-path", type=Path,help="模型参数文件路径")
    parser.add_argument("--vocab-file", type=Path,help="BPE 词表文件路径")
    parser.add_argument("--merges-file", type=Path,help="BPE 合并规则文件路径")
    parser.add_argument("--special-token", dest="special_tokens", action="append", 
                        default=["<|endoftext|>"],help="特殊 Token，可重复指定")
    parser.add_argument("--prompt",help="输入提示文本")
    parser.add_argument("--max-new-tokens", type=int,help="最大生成 Token 数")
    parser.add_argument("--temperature", type=float,help="控制生成分布的随机性")
    parser.add_argument("--top-p", type=float,help="核采样的累积概率阈值")
    parser.add_argument("--vocab-size", type=int,help="	Tokenizer 词表大小")
    parser.add_argument("--context-length", type=int,help="模型最大上下文长度")
    parser.add_argument("--num-layers", type=int,help="	Transformer Decoder 层数")
    parser.add_argument("--d-model", type=int,help="模型隐藏维度")
    parser.add_argument("--d-ff", type=int,help="前馈网络中间层维度")
    parser.add_argument("--num-heads", type=int,help="注意力头数")
    parser.add_argument("--rope-theta", type=float,help="RoPE 基础频率参数")
    parser.add_argument("--device",help="推理设备，例如 cpu、cuda")
    parser.add_argument("--eps-rms1", dest="eps_RMS1", type=float)
    parser.add_argument("--eps-rms2", dest="eps_RMS2", type=float)
    parser.add_argument("--eps-rms3", dest="eps_RMS3", type=float)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> None:
    from ..generation.generator import generate

    model_settings = resolve_settings(
        args.preset,
        args,
        GENERATION_PRESETS,
        required_fields=(
            "params_path", "vocab_file", "merges_file", "prompt", "vocab_size",
            "context_length", "num_layers", "d_model", "d_ff", "num_heads",
            "rope_theta",
        ),
    )
    try:
        tokenizer_config = TokenizerConfig(
            vocab_filepath=model_settings.pop("vocab_file"),
            merges_filepath=model_settings.pop("merges_file"),
            special_tokens=model_settings.pop("special_tokens", []),
        )
        config = GenerationConfig(
            **model_settings,
            tokenizer=tokenizer_config,
        )
    except TypeError as error:
        raise ValueError(f"生成配置不完整：{error}") from error
    print("开始推理：")
    print(generate(config))
    print("推理结束")

