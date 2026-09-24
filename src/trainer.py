"""
Training loop, learning rate scheduling, and checkpointed spectral logging.
"""

import math
import time
import random
import gc
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import PreTrainedTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from tqdm.auto import tqdm

from src.spectral import get_adapter_layers, spectral_snapshot
from src.evaluation import eval_ppl, eval_truthfulqa, eval_arc, effective_rank_truncation_test


def cosine_schedule(optimizer: torch.optim.Optimizer, warmup: int, total: int) -> LambdaLR:
    """Create cosine learning rate schedule with linear warm-up phase."""
    def lr_fn(step: int) -> float:
        if step < warmup:
            return float(step) / float(max(1, warmup))
        p = float(step - warmup) / float(max(1, total - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    return LambdaLR(optimizer, lr_fn)


def logging_schedule(
    early_until: int,
    early_stride: int,
    late_every: int,
    probe_step: int,
    total_steps: int
) -> List[int]:
    """
    Construct non-uniform checkpoint logging schedule prioritizing early optimization dynamics.

    Logs frequently during early training steps, includes the probe step, and continues at coarser intervals.
    """
    steps = set(range(0, early_until + 1, early_stride))
    steps |= set(range(early_until, total_steps + 1, late_every))
    steps.add(probe_step)
    steps.add(total_steps)
    return sorted(s for s in steps if 0 <= s <= total_steps)


def nuclear_norm_penalty(model: nn.Module, coef: float, device: str) -> torch.Tensor:
    """Compute nuclear norm penalty across adapter parameter products as a spectral regularizer."""
    if coef <= 0:
        return torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)
    for _, key, mod in get_adapter_layers(model):
        A = mod.lora_A[key].weight
        B = mod.lora_B[key].weight
        scaling = mod.scaling[key] if hasattr(mod, "scaling") else 1.0
        dW = (B @ A).float() * scaling
        total = total + torch.linalg.matrix_norm(dW, ord="nuc")
    return coef * total


def single_run(
    base_model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    train_ds: Dataset,
    val_ds: Dataset,
    method_label: str,
    rank: int,
    alpha: int,
    dropout: float,
    wd: float,
    use_dora: bool,
    seed: int,
    config: Dict[str, Any],
    device: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Execute a single fine-tuning run with dense spectral tracking and downstream evaluation.

    Args:
        base_model: Pre-trained causal language model.
        tokenizer: PreTrainedTokenizer instance.
        train_ds: Training dataset.
        val_ds: Validation dataset.
        method_label: Descriptive configuration label.
        rank: Nominal rank r.
        alpha: LoRA scaling factor.
        dropout: Adapter dropout probability.
        wd: Weight decay applied to adapter parameters.
        use_dora: Boolean flag enabling DoRA.
        seed: Random seed for initialization and data shuffling.
        config: Full configuration dictionary.
        device: Active compute device.

    Returns:
        Tuple containing run summary results, trajectory rows, SVD rows, TruthfulQA items, ARC items, and truncation rows.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    tag = f"{method_label}_s{seed}"
    print(f"\n  ── {tag} {'─' * 40}")

    peft_kwargs = dict(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=config["_TARGET_MODULES"],
        bias="none",
        inference_mode=False,
    )
    if use_dora:
        try:
            peft_kwargs["use_dora"] = True
        except Exception:
            print("  [Warning] DoRA unsupported in installed PEFT version; falling back to LoRA")
            use_dora = False

    lora_cfg = LoraConfig(**peft_kwargs)
    model = get_peft_model(base_model, lora_cfg)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"     method={method_label} rank={rank} alpha={alpha} params={n_par:,} "
        f"wd={wd} drop={dropout} dora={use_dora}"
    )

    optimizer = AdamW(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=wd)
    warmup_steps = int(config["TRAIN_STEPS"] * config["WARMUP_RATIO"])
    scheduler = cosine_schedule(optimizer, warmup_steps, config["TRAIN_STEPS"])
    loader = DataLoader(train_ds, batch_size=config["BATCH_SIZE"], shuffle=True, drop_last=True)

    log_steps = logging_schedule(
        early_until=config["EARLY_LOG_UNTIL"],
        early_stride=config["EARLY_LOG_STRIDE"],
        late_every=config["LATE_LOG_EVERY"],
        probe_step=config["PROBE_STEP"],
        total_steps=config["TRAIN_STEPS"]
    )

    spectral_rows, svd_rows = [], []
    snap, svd = spectral_snapshot(model, rank, 0, is_dora=use_dora, total_steps=config["TRAIN_STEPS"])
    spectral_rows.extend(snap)
    svd_rows.extend(svd or [])

    model.train()
    step, losses, non_finite_events = 0, [], 0
    time_to_probe_sec = None
    t0 = time.time()
    optimizer.zero_grad()

    pbar = tqdm(total=config["TRAIN_STEPS"], desc=f"train[{tag}]", leave=False)
    while step < config["TRAIN_STEPS"]:
        for batch in loader:
            if step >= config["TRAIN_STEPS"]:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            try:
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                    loss = model(**batch).loss
                    if config["NUCLEAR_NORM_COEF"] > 0:
                        loss = loss + nuclear_norm_penalty(model, config["NUCLEAR_NORM_COEF"], device)
            except Exception:
                loss = model(**batch).loss

            loss_val = loss.item()
            if not math.isfinite(loss_val):
                non_finite_events += 1
            else:
                (loss / config["GRAD_ACCUM"]).backward()
                losses.append(loss_val)

            if (step + 1) % config["GRAD_ACCUM"] == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not math.isfinite(float(gnorm)):
                    non_finite_events += 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step in log_steps:
                snap_t, svd_t = spectral_snapshot(model, rank, step, is_dora=use_dora, total_steps=config["TRAIN_STEPS"])
                spectral_rows.extend(snap_t)
                svd_rows.extend(svd_t or [])

            if step == config["PROBE_STEP"]:
                time_to_probe_sec = time.time() - t0

            step += 1
            pbar.update(1)
    pbar.close()

    train_time = time.time() - t0
    if time_to_probe_sec is None:
        time_to_probe_sec = train_time

    snap, svd = spectral_snapshot(model, rank, config["TRAIN_STEPS"], is_dora=use_dora, total_steps=config["TRAIN_STEPS"])
    spectral_rows.extend(snap)
    svd_rows.extend(svd or [])

    final_ppl = eval_ppl(model, val_ds, device)
    mc1, mc1_items = eval_truthfulqa(model, tokenizer, config["EVAL_TRUTHFUL"], device)
    arc, arc_items = eval_arc(model, tokenizer, config["EVAL_ARC"], device)

    trunc_rows = []
    if config["RUN_TRUNCATION_TEST"]:
        ba_rows = [r for r in snap if r["delta_type"] == "BA" and not r["degenerate"]]
        mean_sr_final = float(np.mean([r["stable_rank"] for r in ba_rows])) if ba_rows else float("nan")
        try:
            trunc_rows = effective_rank_truncation_test(
                model=model,
                val_ds=val_ds,
                nominal_rank=rank,
                mean_sr=mean_sr_final,
                eval_ppl_fn=lambda m, ds: eval_ppl(m, ds, device),
                n_want=config.get("TRUNCATION_K_COUNT")
            )
        except Exception as e:
            print(f"     [Warning] SVD truncation test failed: {e}")

    ba_final = [r for r in snap if r["delta_type"] == "BA" and not r["degenerate"]]
    result = {
        "model": config["_SHORT"],
        "method": method_label,
        "rank": rank,
        "alpha": alpha,
        "alpha_schedule": config["ALPHA_SCHEDULE"],
        "seed": seed,
        "dropout": dropout,
        "weight_decay": wd,
        "dora": use_dora,
        "use_4bit": config["USE_4BIT"],
        "target_scope": config["TARGET_SCOPE"],
        "loss_scope": config["LOSS_SCOPE"],
        "n_params": n_par,
        "final_ppl": final_ppl,
        "truthfulqa_mc1": mc1 if mc1 is not None else -1,
        "arc_easy": arc if arc is not None else -1,
        "n_truthfulqa_scored": len(mc1_items),
        "n_arc_scored": len(arc_items),
        "train_loss_last30": round(float(np.mean(losses[-30:])), 4) if losses else float("nan"),
        "non_finite_events": non_finite_events,
        "mean_sr_final": round(float(np.mean([r["stable_rank"] for r in ba_final])), 4) if ba_final else float("nan"),
        "min_sr_final": round(float(np.min([r["stable_rank"] for r in ba_final])), 4) if ba_final else float("nan"),
        "max_sr_final": round(float(np.max([r["stable_rank"] for r in ba_final])), 4) if ba_final else float("nan"),
        "mean_A_fro_final": round(float(np.mean([r["A_fro"] for r in ba_final])), 4) if ba_final else float("nan"),
        "mean_B_fro_final": round(float(np.mean([r["B_fro"] for r in ba_final])), 4) if ba_final else float("nan"),
        "train_min": round(train_time / 60, 2),
        "time_to_probe_min": round(time_to_probe_sec / 60, 2),
    }

    print(
        f"     PPL={final_ppl:.2f}  MC1={mc1}  ARC={arc}  "
        f"mean_sr={result['mean_sr_final']}  non_finite={non_finite_events}  "
        f"{train_time / 60:.1f} min"
    )

    for r in mc1_items:
        r.update({"model": config["_SHORT"], "method": method_label, "seed": seed, "rank": rank, "task": "truthfulqa"})
    for r in arc_items:
        r.update({"model": config["_SHORT"], "method": method_label, "seed": seed, "rank": rank, "task": "arc_easy"})
    for r in trunc_rows:
        r.update({"model": config["_SHORT"], "method": method_label, "seed": seed})

    try:
        model.delete_adapter("default")
    except Exception:
        pass
    del model, optimizer, scheduler
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return result, spectral_rows, svd_rows, mc1_items, arc_items, trunc_rows
