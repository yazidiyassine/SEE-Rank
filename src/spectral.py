"""
Spectral diagnostic utilities for LoRA and DoRA adapters.
Computes exact SVD, stable rank, effective rank, and energy concentration.
"""

import math
import importlib
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn


ATTN_LEAF_CANDIDATES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "dense", "out_proj",
    "Wqkv", "c_attn", "c_proj", "query_key_value", "qkv_proj",
}

MLP_LEAF_CANDIDATES = {
    "gate_proj", "up_proj", "down_proj", "fc1", "fc2",
    "dense_h_to_4h", "dense_4h_to_h", "w1", "w2", "w3",
}


def get_peft_builtin_target_map() -> Dict[str, List[str]]:
    """Retrieve default target module mappings from installed PEFT library."""
    for modpath in ("peft.utils.constants", "peft.utils.other", "peft.utils"):
        try:
            mod = importlib.import_module(modpath)
            mapping = getattr(mod, "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING", None)
            if mapping:
                return mapping
        except Exception:
            continue
    return {}


def introspect_linear_leaves(model: nn.Module, candidates: set) -> List[str]:
    """Find linear layer names matching candidate sets within the model architecture."""
    found = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in candidates:
                found.add(leaf)
    return sorted(found)


def resolve_target_modules(model: nn.Module, scope: str = "attn") -> List[str]:
    """
    Identify target projection modules for adapter placement based on model architecture.

    Args:
        model: Base causal language model.
        scope: Target scope, either 'attn' (attention projections) or 'attn_mlp' (attention + MLP).

    Returns:
        List of module leaf names to adapt.
    """
    model_type = getattr(model.config, "model_type", None)
    builtin = get_peft_builtin_target_map()
    modules = list(builtin.get(model_type, []) or [])

    if scope == "attn" and modules:
        attn_only = [m for m in modules if m in ATTN_LEAF_CANDIDATES]
        modules = attn_only if attn_only else []

    if not modules:
        modules = introspect_linear_leaves(model, ATTN_LEAF_CANDIDATES)

    if not modules:
        modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    if scope == "attn_mlp":
        mlp = introspect_linear_leaves(model, MLP_LEAF_CANDIDATES)
        mlp = sorted(set(mlp) | {m for m in (builtin.get(model_type, []) or []) if m in MLP_LEAF_CANDIDATES})
        modules = sorted(set(modules) | set(mlp))

    return modules


def count_adapted_leaves(model: nn.Module, target_modules: List[str]) -> int:
    """Count the total number of linear layers targeted for adapter attachment."""
    return sum(
        1 for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in target_modules
    )


def alpha_for_rank(r: int, schedule: str, fixed_value: int = 16) -> int:
    """
    Calculate adapter scaling factor alpha based on the specified schedule.

    Args:
        r: Nominal adapter rank.
        schedule: Schedule type ('fixed', '2r', or 'sqrt').
        fixed_value: Fixed alpha value when schedule is 'fixed'.

    Returns:
        Integer scaling factor alpha.
    """
    if schedule == "fixed":
        return fixed_value
    if schedule == "2r":
        return 2 * r
    if schedule == "sqrt":
        return max(1, round(4 * math.sqrt(r)))
    raise ValueError(f"Unknown alpha schedule: {schedule}")


@torch.no_grad()
def get_adapter_layers(model: nn.Module) -> List[Tuple[str, str, nn.Module]]:
    """Extract all active LoRA layers along with layer names and adapter keys."""
    out = []
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            for key in mod.lora_A.keys():
                parts = name.replace("base_model.model.", "").split(".")
                short_name = ".".join(parts[-2:]) if len(parts) >= 2 else name
                out.append((short_name, key, mod))
    return out


@torch.no_grad()
def dora_merged_delta(mod: nn.Module, key: str) -> Optional[torch.Tensor]:
    """
    Compute effective weight update for DoRA layers including magnitude scaling.

    Args:
        mod: Target adapted module.
        key: Adapter identifier key.

    Returns:
        Tensor of effective weight delta W' - W0, or None if not applicable.
    """
    try:
        mag_dict = getattr(mod, "lora_magnitude_vector", None)
        if mag_dict is None or key not in mag_dict:
            return None
        magnitude = mag_dict[key]
        if hasattr(magnitude, "weight"):
            magnitude = magnitude.weight
        magnitude = magnitude.float().flatten()

        base_layer = mod.get_base_layer() if hasattr(mod, "get_base_layer") else mod.base_layer
        W0 = base_layer.weight.float()
        A = mod.lora_A[key].weight.float()
        B = mod.lora_B[key].weight.float()
        scaling = mod.scaling[key] if hasattr(mod, "scaling") else 1.0
        BA = (B @ A) * scaling

        directional = W0 + BA
        col_norm = directional.norm(dim=1, keepdim=True).clamp_min(1e-8)
        W_new = directional * (magnitude.unsqueeze(1) / col_norm)
        return W_new - W0
    except Exception:
        return None


@torch.no_grad()
def spectral_snapshot(
    model: nn.Module,
    nominal_rank: int,
    step: int,
    is_dora: bool = False,
    total_steps: int = 40
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Compute exact singular value decomposition and spectral diagnostics across adapter layers.

    Args:
        model: Adapted neural network.
        nominal_rank: Nominal rank parameter r.
        step: Current training step.
        is_dora: Flag indicating whether DoRA magnitude decomposition is enabled.
        total_steps: Total training steps for logging final spectra.

    Returns:
        Tuple containing:
            - layer_rows: Summary spectral metrics per layer (stable rank, top-1 energy, utilization).
            - svd_rows: Individual singular values at boundary steps (0 and terminal).
    """
    layer_rows, svd_rows = [], []

    for short_name, key, mod in get_adapter_layers(model):
        A = mod.lora_A[key].weight
        B = mod.lora_B[key].weight
        scaling = mod.scaling[key] if hasattr(mod, "scaling") else 1.0

        dW_ba = (B @ A).float() * scaling
        frob_sq_ba = torch.sum(dW_ba ** 2).item()

        def _metrics_from(dW, frob_sq, delta_type):
            if frob_sq < 1e-16:
                return {
                    "step": step,
                    "layer": short_name,
                    "delta_type": delta_type,
                    "stable_rank": float("nan"),
                    "top1_energy": float("nan"),
                    "rank_util_pct": float("nan"),
                    "frob_norm": 0.0,
                    "degenerate": True,
                    "A_fro": torch.linalg.norm(A).item(),
                    "B_fro": torch.linalg.norm(B).item(),
                }, None

            S = torch.linalg.svdvals(dW)
            S_sq = S ** 2
            total_e = S_sq.sum().item()
            sr = total_e / (S_sq[0].item() + 1e-16)

            row = {
                "step": step,
                "layer": short_name,
                "delta_type": delta_type,
                "stable_rank": round(sr, 6),
                "top1_energy": round(S_sq[0].item() / (total_e + 1e-16), 6),
                "rank_util_pct": round((sr / nominal_rank) * 100.0, 4),
                "frob_norm": round(math.sqrt(total_e), 6),
                "degenerate": False,
                "A_fro": torch.linalg.norm(A).item(),
                "B_fro": torch.linalg.norm(B).item(),
            }

            sv_rows = None
            if step == 0 or step >= total_steps:
                sv_rows = [
                    {
                        "step": step,
                        "layer": short_name,
                        "delta_type": delta_type,
                        "sv_index": i,
                        "sv_value": round(v, 8)
                    }
                    for i, v in enumerate(S[:nominal_rank].tolist())
                ]
            return row, sv_rows

        row_ba, sv_ba = _metrics_from(dW_ba, frob_sq_ba, "BA")
        layer_rows.append(row_ba)
        if sv_ba:
            svd_rows.extend(sv_ba)

        if is_dora:
            dW_dora = dora_merged_delta(mod, key)
            if dW_dora is not None:
                frob_sq_dora = torch.sum(dW_dora ** 2).item()
                row_dora, sv_dora = _metrics_from(dW_dora, frob_sq_dora, "DoRA_merged")
                layer_rows.append(row_dora)
                if sv_dora:
                    svd_rows.extend(sv_dora)

    return layer_rows, svd_rows
