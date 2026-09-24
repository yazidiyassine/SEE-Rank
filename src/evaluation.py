"""
Evaluation suite for perplexity, multiple-choice downstream tasks, and SVD truncation testing.
"""

import math
from typing import List, Tuple, Dict, Any, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer
from tqdm.auto import tqdm

from src.spectral import get_adapter_layers


@torch.no_grad()
def eval_ppl(model: nn.Module, val_ds: Dataset, device: str) -> float:
    """
    Compute sequence cross-entropy perplexity over validation dataset.

    Args:
        model: Evaluated neural network.
        val_ds: Validation dataset formatted with input_ids and labels.
        device: Active compute device string.

    Returns:
        Perplexity value rounded to 4 decimal places.
    """
    was_training = model.training
    model.eval()
    loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    total_loss, total_tok = 0.0, 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        n = (batch["labels"] != -100).sum().item()
        total_loss += out.loss.item() * n
        total_tok += n

    if was_training:
        model.train()

    avg_loss = total_loss / max(total_tok, 1)
    return round(math.exp(min(avg_loss, 20.0)), 4)


@torch.no_grad()
def _mc_eval(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    n_samples: int,
    question_key: str,
    choices_fn: Callable[[Dict[str, Any]], List[str]],
    correct_fn: Callable[[Dict[str, Any]], Optional[int]],
    prompt_fn: Callable[[str, str], str],
    desc: str,
    device: str
) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    """Internal helper evaluating multiple-choice accuracy via normalized sequence loss."""
    try:
        ds = load_dataset(dataset_name, dataset_config, split=split)
    except Exception as e:
        print(f"  [Warning] Could not load {dataset_name}/{dataset_config}: {e}")
        return None, []

    ds = ds.select(range(min(n_samples, len(ds))))
    was_training = model.training
    model.eval()
    per_item = []

    for item_idx, item in enumerate(tqdm(ds, desc=desc, leave=False, unit="q")):
        try:
            question = item[question_key]
            choices = choices_fn(item)
            gold = correct_fn(item)
            if gold is None or gold < 0 or gold >= len(choices):
                continue
        except Exception:
            continue

        prompts = [prompt_fn(question, c) for c in choices]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=96,
            padding=True
        ).to(device)

        logits = model(**inputs).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs["input_ids"][:, 1:].contiguous()
        mask = inputs["attention_mask"][:, 1:].contiguous()

        tok_loss = nn.CrossEntropyLoss(reduction="none")(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_logits.size(0), -1)

        seq_loss = (tok_loss * mask.float()).sum(-1) / mask.sum(-1).clamp(min=1)
        pred = int(seq_loss.argmin())
        per_item.append({"item_index": item_idx, "correct": int(pred == gold)})

    if was_training:
        model.train()

    n_scored = len(per_item)
    n_correct = sum(r["correct"] for r in per_item)
    acc = n_correct / max(n_scored, 1)
    print(f"  {desc}: {acc:.4f} ({n_correct}/{n_scored})")
    return round(acc, 4), per_item


def eval_truthfulqa(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    n_samples: int,
    device: str
) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    """Evaluate TruthfulQA MC1 benchmark accuracy."""
    return _mc_eval(
        model=model,
        tokenizer=tokenizer,
        dataset_name="truthful_qa",
        dataset_config="multiple_choice",
        split="validation",
        n_samples=n_samples,
        question_key="question",
        choices_fn=lambda it: it["mc1_targets"]["choices"],
        correct_fn=lambda it: it["mc1_targets"]["labels"].index(1) if 1 in it["mc1_targets"]["labels"] else None,
        prompt_fn=lambda q, c: f"Q: {q}\nA: {c}",
        desc="TruthfulQA",
        device=device
    )


def eval_arc(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    n_samples: int,
    device: str
) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    """Evaluate ARC-Easy multiple-choice benchmark accuracy."""
    return _mc_eval(
        model=model,
        tokenizer=tokenizer,
        dataset_name="allenai/ai2_arc",
        dataset_config="ARC-Easy",
        split="test",
        n_samples=n_samples,
        question_key="question",
        choices_fn=lambda it: it["choices"]["text"],
        correct_fn=lambda it: it["choices"]["label"].index(it["answerKey"]) if it["answerKey"] in it["choices"]["label"] else None,
        prompt_fn=lambda q, c: f"Question: {q}\nAnswer: {c}",
        desc="ARC-Easy",
        device=device
    )


@torch.no_grad()
def effective_rank_truncation_test(
    model: nn.Module,
    val_ds: Dataset,
    nominal_rank: int,
    mean_sr: float,
    eval_ppl_fn: Callable[[nn.Module, Dataset], float],
    n_want: Optional[int] = 2
) -> List[Dict[str, Any]]:
    """
    Assess construct validity of stable rank by truncating trained adapters to top k singular components.

    Replaces adapter weights in-place with low-rank approximations, computes validation perplexity,
    and restores original adapter parameters.

    Args:
        model: Adapted neural network.
        val_ds: Held-out validation dataset.
        nominal_rank: Nominal allocated rank r.
        mean_sr: Empirical mean stable rank of adapter layers.
        eval_ppl_fn: Callable computing validation perplexity.
        n_want: Subsampling count for truncation ranks (default: 2 for k=1 and k=r).

    Returns:
        List of dictionaries recording perplexity across tested truncation ranks k.
    """
    layers = get_adapter_layers(model)
    if not layers:
        return []

    ks = sorted(set(
        int(k) for k in [
            1,
            2,
            max(1, math.ceil(mean_sr)) if not math.isnan(mean_sr) else 2,
            max(1, nominal_rank // 2),
            nominal_rank
        ]
        if 1 <= k <= nominal_rank
    ))

    if n_want and len(ks) > n_want:
        if n_want == 1:
            ks = [ks[-1]]
        else:
            idx = np.linspace(0, len(ks) - 1, n_want).round().astype(int)
            ks = sorted(set(ks[i] for i in idx))

    originals = {}
    for short_name, key, mod in layers:
        originals[(short_name, key)] = (
            mod.lora_A[key].weight.data.clone(),
            mod.lora_B[key].weight.data.clone(),
        )

    rows = []
    try:
        for k in ks:
            for short_name, key, mod in layers:
                A0, B0 = originals[(short_name, key)]
                scaling = mod.scaling[key] if hasattr(mod, "scaling") else 1.0
                dW = (B0.float() @ A0.float()) * scaling

                U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
                kk = min(k, S.shape[0])
                sqrtS = S[:kk].clamp_min(0).sqrt()
                Bk = U[:, :kk] * sqrtS
                Ak = sqrtS.unsqueeze(1) * Vh[:kk, :]

                denom = math.sqrt(scaling) if scaling > 0 else 1.0
                mod.lora_B[key].weight.data = (Bk / denom).to(B0.dtype)
                mod.lora_A[key].weight.data = (Ak / denom).to(A0.dtype)

            ppl_k = eval_ppl_fn(model, val_ds)
            rows.append({
                "k": k,
                "nominal_rank": nominal_rank,
                "val_ppl_at_k": ppl_k
            })
    finally:
        for short_name, key, mod in layers:
            A0, B0 = originals[(short_name, key)]
            mod.lora_A[key].weight.data = A0
            mod.lora_B[key].weight.data = B0

    return rows
