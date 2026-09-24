"""
SEE-Rank: Spectral Early-Exit Rank Selection algorithm and baseline evaluation.
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd


def see_rank_select(
    ranks_sorted: List[int],
    sr_at_probe: Dict[int, float],
    tau: float
) -> Tuple[Optional[int], List[Dict[str, Any]]]:
    """
    Execute SEE-Rank probe-and-stop decision rule across evaluated ranks.

    Iterates through candidate ranks in increasing order. Stops and selects rank r_{k-1}
    when the marginal spectral gain g_k = (sr_k / sr_{k-1}) - 1 drops below threshold tau.

    Args:
        ranks_sorted: Sorted list of candidate ranks.
        sr_at_probe: Dictionary mapping rank to empirical mean stable rank measured at probe step.
        tau: Marginal gain threshold parameter.

    Returns:
        Tuple containing selected rank and trace list of decision steps.
    """
    trace = []
    valid_ranks = [r for r in ranks_sorted if r in sr_at_probe and not math.isnan(sr_at_probe[r])]
    if len(valid_ranks) < 2:
        return (valid_ranks[0] if valid_ranks else None), trace

    for i in range(1, len(valid_ranks)):
        r_prev, r_cur = valid_ranks[i - 1], valid_ranks[i]
        sr_prev, sr_cur = sr_at_probe[r_prev], sr_at_probe[r_cur]
        g = (sr_cur / sr_prev - 1.0) if sr_prev > 0 else float("inf")
        stop = g < tau
        trace.append({
            "r_prev": r_prev,
            "r_cur": r_cur,
            "gain": g,
            "tau": tau,
            "stop": stop
        })
        if stop:
            return r_prev, trace

    return valid_ranks[-1], trace


def run_see_rank_analysis(
    df_results: pd.DataFrame,
    df_spectral: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate SEE-Rank across a sensitivity grid of tau thresholds and compare with baselines.

    Baselines compared:
        - Full rank sweep (ground-truth best rank by validation perplexity).
        - Fixed default rank baseline (e.g. r=16).
        - Short validation-loss sweep baseline.

    Reports end-to-end computational costs:
        Probe cost for all candidates + completion cost of the selected candidate.

    Args:
        df_results: DataFrame containing run-level summary metrics.
        df_spectral: DataFrame containing checkpoint-level spectral logs.
        config: Experiment configuration dictionary.

    Returns:
        Tuple of (see_rank_summary_df, see_rank_traces_df).
    """
    base_rows = df_results[
        df_results["method"].str.startswith("LoRA_r") &
        ~df_results["method"].str.contains("wd|drop")
    ].copy()

    if base_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    ranks_sorted = sorted(base_rows["rank"].unique().tolist())

    probe_spec = df_spectral[
        (df_spectral["step"] == config["PROBE_STEP"]) &
        (df_spectral["delta_type"] == "BA") &
        (~df_spectral["degenerate"])
    ]
    sr_at_probe = probe_spec.groupby("rank")["stable_rank"].mean().to_dict()

    full_ppl = base_rows.groupby("rank")["final_ppl"].mean()
    best_rank_full_sweep = int(full_ppl.idxmin())

    probe_cost_total = base_rows.groupby("rank")["time_to_probe_min"].mean().sum()
    completion_cost = (
        base_rows.groupby("rank")["train_min"].mean() -
        base_rows.groupby("rank")["time_to_probe_min"].mean()
    )
    full_sweep_cost_total = base_rows.groupby("rank")["train_min"].mean().sum()
    default_rank_cost = base_rows.groupby("rank")["train_min"].mean().get(
        config["SEE_RANK_DEFAULT_RANK"], float("nan")
    )

    rows, traces = [], []
    for tau in config["SEE_RANK_TAU_GRID"]:
        sel_rank, trace = see_rank_select(ranks_sorted, sr_at_probe, tau)
        for t in trace:
            t["tau"] = tau
            traces.append(t)

        completion_time = completion_cost.get(sel_rank, float("nan")) if sel_rank else float("nan")
        end_to_end_cost = probe_cost_total + completion_time

        rows.append({
            "tau": tau,
            "selected_rank_see_rank": sel_rank,
            "selected_rank_val_loss_sweep": int(
                base_rows[base_rows["rank"].isin(ranks_sorted)]
                .groupby("rank")["final_ppl"].mean().idxmin()
            ),
            "default_rank_baseline": config["SEE_RANK_DEFAULT_RANK"],
            "best_rank_full_sweep": best_rank_full_sweep,
            "see_rank_matches_full_sweep": sel_rank == best_rank_full_sweep,
            "end_to_end_cost_min": round(end_to_end_cost, 2),
            "full_sweep_cost_min": round(full_sweep_cost_total, 2),
            "default_rank_cost_min": (
                round(float(default_rank_cost), 2)
                if not math.isnan(default_rank_cost)
                else float("nan")
            ),
        })

    return pd.DataFrame(rows), pd.DataFrame(traces)
