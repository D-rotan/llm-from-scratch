from dataclasses import dataclass
from pathlib import Path

@dataclass
class BPETrainerConfig:
    input_path: Path
    vocab_size: int
    special_tokens: list[str]
    num_processes: int | None
    outputs: Path

@dataclass
class TokenizerConfig:
    vocab_filepath: Path
    merges_filepath: Path | None
    special_tokens: list[str]

@dataclass
class EncodeRequest:
    text:str

@dataclass
class TrainRequest:
    text_path:Path
    save_path:Path

@dataclass
class DecodeRequest:
    ids:list[int]

@dataclass
class ModelTrainerConfig:
    src: Path
    max_iters: int
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    d_ff: int
    num_heads: int
    rope_theta: float
    batch_size: int
    device: str
    lr_max: float
    lr_min: float
    warmup_iters: int
    cosine_cycle_iters: int
    max_l2_norm: float
    checkpoint_iters: int
    out: Path
    lr: float = 1e-3
    on_gradient_clipping: bool = True
    on_learning_rate_schedule: bool = True
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 1e-3
    eps_RMS1: float = 1e-5
    eps_RMS2: float = 1e-5
    eps_RMS3: float = 1e-5
    eps_AdamW: float = 1e-8
    eps_gradient: float = 1e-6


@dataclass
class GenerationConfig:
    params_path: Path
    tokenizer: TokenizerConfig
    prompt: str
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    d_ff: int
    num_heads: int
    rope_theta: float
    device: str = "cpu"
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_p: float = 0.5
    eps_RMS1: float = 1e-5
    eps_RMS2: float = 1e-5
    eps_RMS3: float = 1e-5

