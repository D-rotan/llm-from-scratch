from pathlib import Path


MODEL_PRESETS = {
    "small": {
        "src": Path("data/tokenizer.npy"),
        "max_iters": 1_000,
        "vocab_size": 10_000,
        "context_length": 128,
        "num_layers": 2,
        "d_model": 128,
        "d_ff": 512,
        "num_heads": 4,
        "rope_theta": 10_000.0,
        "batch_size": 4,
        "device": "cpu",
        "lr_max": 1e-3,
        "lr_min": 1e-4,
        "warmup_iters": 100,
        "cosine_cycle_iters": 900,
        "max_l2_norm": 1.0,
        "checkpoint_iters": 100,
        "out": Path("data/model_params/small.pt"),
        "lr": 1e-3,
        "on_gradient_clipping": True,
        "on_learning_rate_schedule": True,
        "betas": (0.9, 0.99),
        "weight_decay": 1e-3,
        "eps_RMS1": 1e-5,
        "eps_RMS2": 1e-5,
        "eps_RMS3": 1e-5,
        "eps_AdamW": 1e-8,
        "eps_gradient": 1e-6,
    },
    "mid": {
        "src": Path("data/tokenizer.npy"),
        "max_iters": 10_000,
        "vocab_size": 15_000,
        "context_length": 256,
        "num_layers": 6,
        "d_model": 384,
        "d_ff": 1_536,
        "num_heads": 6,
        "rope_theta": 10_000.0,
        "batch_size": 8,
        "device": "cpu",
        "lr_max": 3e-4,
        "lr_min": 3e-5,
        "warmup_iters": 1_000,
        "cosine_cycle_iters": 9_000,
        "max_l2_norm": 1.0,
        "checkpoint_iters": 500,
        "out": Path("data/model_params/mid.pt"),
        "lr": 3e-4,
        "on_gradient_clipping": True,
        "on_learning_rate_schedule": True,
        "betas": (0.9, 0.99),
        "weight_decay": 1e-3,
        "eps_RMS1": 1e-5,
        "eps_RMS2": 1e-5,
        "eps_RMS3": 1e-5,
        "eps_AdamW": 1e-8,
        "eps_gradient": 1e-6,
    },
    "large": {
        "src": Path("data/tokenizer.npy"),
        "max_iters": 100_000,
        "vocab_size": 24_000,
        "context_length": 512,
        "num_layers": 12,
        "d_model": 768,
        "d_ff": 3_072,
        "num_heads": 12,
        "rope_theta": 10_000.0,
        "batch_size": 8,
        "device": "cpu",
        "lr_max": 1e-4,
        "lr_min": 1e-5,
        "warmup_iters": 2_000,
        "cosine_cycle_iters": 98_000,
        "max_l2_norm": 1.0,
        "checkpoint_iters": 1_000,
        "out": Path("data/model_params/large.pt"),
        "lr": 1e-4,
        "on_gradient_clipping": True,
        "on_learning_rate_schedule": True,
        "betas": (0.9, 0.99),
        "weight_decay": 1e-3,
        "eps_RMS1": 1e-5,
        "eps_RMS2": 1e-5,
        "eps_RMS3": 1e-5,
        "eps_AdamW": 1e-8,
        "eps_gradient": 1e-6,
    },
}


GENERATION_PRESETS = {
    name: {
        "params_path": values["out"],
        "vocab_file": Path("data/tokenizer/vocab.pkl"),
        "merges_file": Path("data/tokenizer/merges.pkl"),
        "special_tokens": [],
        "prompt": "你好",
        "max_new_tokens": 50,
        "temperature": 1.0,
        "top_p": 0.9,
        "vocab_size": values["vocab_size"],
        "context_length": values["context_length"],
        "num_layers": values["num_layers"],
        "d_model": values["d_model"],
        "d_ff": values["d_ff"],
        "num_heads": values["num_heads"],
        "rope_theta": values["rope_theta"],
        "device": values["device"],
        "eps_RMS1": values["eps_RMS1"],
        "eps_RMS2": values["eps_RMS2"],
        "eps_RMS3": values["eps_RMS3"],
    }
    for name, values in MODEL_PRESETS.items()
}


def resolve_settings(
    preset: str | None,
    args,
    presets: dict[str, dict],
    required_fields: tuple[str, ...] = (),
) -> dict:
    settings = presets[preset].copy() if preset is not None else {}
    fields = set().union(*(values.keys() for values in presets.values()))
    for field in fields:
        value = getattr(args, field, None)
        if value is not None:
            settings[field] = value

    if preset is None:
        missing = sorted(field for field in required_fields if field not in settings)
        if missing:
            names = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            raise ValueError(f"未选择 preset，必须提供：{names}")
    return settings
