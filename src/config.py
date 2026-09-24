"""
Configuration and parameter settings for LoRA spectral analysis experiments.
"""

import os
import argparse
from typing import Dict, Any


DEFAULT_CONFIG: Dict[str, Any] = {
    # Model specification
    "MODEL": "Qwen/Qwen2.5-1.5B",
    "SEEDS": [42, 123],
    "RANKS": [4, 8, 16, 32, 64],

    # LoRA scaling schedule
    "ALPHA_SCHEDULE": "2r",             # Options: "fixed", "2r", "sqrt"
    "ALPHA_FIXED_VALUE": 16,            # Used when ALPHA_SCHEDULE == "fixed"

    # Quantization and module settings
    "USE_4BIT": True,                   # NF4 4-bit base model quantization
    "TARGET_SCOPE": "attn",             # Options: "attn", "attn_mlp"

    # Regularization parameters
    "RUN_REGULARIZATION_CONDITIONS": True,
    "WD_STRONG": 0.10,                  # Weight decay parameter for adapter weights
    "DROPOUT_HIGH": 0.20,               # High-dropout test condition
    "NUCLEAR_NORM_COEF": 0.0,           # Nuclear norm regularization coefficient

    # Optimization budget and checkpointing
    "TRAIN_STEPS": 40,
    "PROBE_STEP": 10,                   # Checkpoint step for early-exit evaluation
    "EARLY_LOG_STRIDE": 2,              # Checkpoint frequency during early phase
    "EARLY_LOG_UNTIL": 20,              # Step boundary for dense logging
    "LATE_LOG_EVERY": 10,               # Checkpoint frequency during late phase

    # Data parameters
    "TRAIN_SAMPLES": 2000,
    "VAL_SAMPLES": 80,
    "MAX_SEQ_LEN": 256,
    "BATCH_SIZE": 4,
    "GRAD_ACCUM": 4,
    "LEARNING_RATE": 2e-4,
    "WARMUP_RATIO": 0.1,
    "DATA_SEED": 1234,
    "LOSS_SCOPE": "full_sequence",      # Options: "full_sequence", "response_only"

    # Downstream evaluation parameters
    "EVAL_TRUTHFUL": 40,
    "EVAL_ARC": 40,

    # SEE-Rank evaluation parameters
    "SEE_RANK_TAU_GRID": [0.10, 0.15, 0.20, 0.25, 0.30],
    "SEE_RANK_DEFAULT_RANK": 16,

    # Truncation construct validity test
    "RUN_TRUNCATION_TEST": True,
    "TRUNCATION_K_COUNT": 2,            # Number of truncation ranks evaluated (e.g. 1 and r)

    # Hardware & output
    "GPU_INDEX": 0,
    "OUT_DIR": "./results",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for experiment configuration."""
    parser = argparse.ArgumentParser(
        description="Spectral Diagnostic for LoRA Adapter Rank Utilization"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Hugging Face model identifier (e.g., Qwen/Qwen2.5-1.5B, meta-llama/Llama-3.2-1B, microsoft/phi-2)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Random seeds to evaluate",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=None,
        help="Rank grid values (e.g., 4 8 16 32 64)",
    )
    parser.add_argument(
        "--alpha-schedule",
        type=str,
        choices=["fixed", "2r", "sqrt"],
        default=None,
        help="Scaling schedule for LoRA alpha",
    )
    parser.add_argument(
        "--target-scope",
        type=str,
        choices=["attn", "attn_mlp"],
        default=None,
        help="Scope of target linear layers to adapt",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit NF4 quantization and train with 16-bit precision",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help="Total training steps per configuration",
    )
    parser.add_argument(
        "--probe-step",
        type=int,
        default=None,
        help="Probe checkpoint step for SEE-Rank selection",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for generated CSV and JSON files",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Physical GPU device index to pin execution to",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run extended benchmark schedule (200 steps, 3 seeds, 200 val samples)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast verification schedule (40 steps, 2 seeds)",
    )
    return parser.parse_args()


def get_config(args: argparse.Namespace = None) -> Dict[str, Any]:
    """Build configuration dictionary merged with command-line overrides."""
    config = dict(DEFAULT_CONFIG)

    if args is None:
        args = parse_args()

    if args.full:
        config.update(
            SEEDS=[42, 123, 456],
            TRAIN_STEPS=200,
            PROBE_STEP=50,
            EARLY_LOG_STRIDE=5,
            EARLY_LOG_UNTIL=100,
            LATE_LOG_EVERY=50,
            VAL_SAMPLES=200,
            EVAL_TRUTHFUL=150,
            EVAL_ARC=200,
            TRUNCATION_K_COUNT=None,
        )
    elif args.fast:
        config.update(
            SEEDS=[42, 123],
            TRAIN_STEPS=40,
            PROBE_STEP=10,
            EARLY_LOG_STRIDE=2,
            EARLY_LOG_UNTIL=20,
            LATE_LOG_EVERY=10,
            VAL_SAMPLES=80,
            EVAL_TRUTHFUL=40,
            EVAL_ARC=40,
            TRUNCATION_K_COUNT=2,
        )

    if args.model:
        config["MODEL"] = args.model
    if args.seeds:
        config["SEEDS"] = args.seeds
    if args.ranks:
        config["RANKS"] = args.ranks
    if args.alpha_schedule:
        config["ALPHA_SCHEDULE"] = args.alpha_schedule
    if args.target_scope:
        config["TARGET_SCOPE"] = args.target_scope
    if args.no_4bit:
        config["USE_4BIT"] = False
    if args.train_steps is not None:
        config["TRAIN_STEPS"] = args.train_steps
    if args.probe_step is not None:
        config["PROBE_STEP"] = args.probe_step
    if args.out_dir:
        config["OUT_DIR"] = args.out_dir
    if args.gpu is not None:
        config["GPU_INDEX"] = args.gpu

    os.makedirs(config["OUT_DIR"], exist_ok=True)
    return config
