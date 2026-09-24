"""
Statistical modeling, functional-form fitting, and hypothesis testing utilities.
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.spectral import alpha_for_rank


def fit_functional_forms(ranks: List[float], srs: List[float]) -> Dict[str, Dict[str, float]]:
    """
    Fit power-law, logarithmic, and saturating models to rank scaling data and compare via AIC.

    Models:
        - Power-law: sr(r) = a * r^beta
        - Logarithmic: sr(r) = a * ln(r) + b
        - Saturating: sr(r) = S * r / (r + r0)

    Args:
        ranks: List of nominal rank values.
        srs: List of corresponding mean stable rank values.

    Returns:
        Dictionary mapping model names to estimated parameters and Akaike Information Criterion (AIC).
    """
    r_arr = np.array(ranks, dtype=float)
    s_arr = np.array(srs, dtype=float)
    mask = np.isfinite(r_arr) & np.isfinite(s_arr) & (s_arr > 0) & (r_arr > 0)
    r_arr, s_arr = r_arr[mask], s_arr[mask]

    out = {}
    if len(r_arr) < 3:
        return out

    def compute_aic(resid: np.ndarray, k_params: int) -> float:
        n = len(resid)
        rss = float(np.sum(resid ** 2))
        if rss <= 0 or n <= k_params:
            return float("inf")
        return n * math.log(rss / n) + 2 * k_params

    # 1. Power-law: log(sr) = log(a) + beta * log(r)
    try:
        logr, logsr = np.log(r_arr), np.log(s_arr)
        A = np.vstack([logr, np.ones_like(logr)]).T
        beta, loga = np.linalg.lstsq(A, logsr, rcond=None)[0]
        resid = logsr - (A @ [beta, loga])
        out["power_law"] = {
            "beta": float(beta),
            "a": float(math.exp(loga)),
            "aic": compute_aic(resid, 2)
        }
    except Exception:
        pass

    # 2. Logarithmic: sr = a * log(r) + b
    try:
        logr = np.log(r_arr)
        A = np.vstack([logr, np.ones_like(logr)]).T
        a, b = np.linalg.lstsq(A, s_arr, rcond=None)[0]
        resid = s_arr - (A @ [a, b])
        out["logarithmic"] = {
            "a": float(a),
            "b": float(b),
            "aic": compute_aic(resid, 2)
        }
    except Exception:
        pass

    # 3. Saturating: sr = S * r / (r + r0)
    try:
        best = None
        for r0 in np.linspace(0.1, r_arr.max() * 3.0, 200):
            x = r_arr / (r_arr + r0)
            dot_xx = float(np.dot(x, x))
            S = float(np.dot(x, s_arr) / dot_xx) if dot_xx > 0 else 0.0
            resid = s_arr - S * x
            aic_val = compute_aic(resid, 2)
            if best is None or aic_val < best[0]:
                best = (aic_val, S, r0)
        if best:
            out["saturating"] = {
                "S": best[1],
                "r0": best[2],
                "aic": best[0]
            }
    except Exception:
        pass

    return out


def seed_level_correlation(
    df_results: pd.DataFrame,
    x_col: str,
    y_col: str
) -> Optional[Dict[str, Any]]:
    """
    Compute Pearson correlation at the seed level across individual experimental runs.

    Args:
        df_results: Results DataFrame containing run metrics.
        x_col: First variable column name.
        y_col: Second variable column name.

    Returns:
        Dictionary with sample size n, correlation coefficient r, and two-sided p-value.
    """
    valid = df_results[(df_results[x_col].notna()) & (df_results[y_col] >= 0)]
    if len(valid) < 4:
        return None
    r, p = stats.pearsonr(valid[x_col], valid[y_col])
    return {
        "x": x_col,
        "y": y_col,
        "n": len(valid),
        "r": round(float(r), 4),
        "p": round(float(p), 6)
    }


def tost_equivalence(
    sample_a: List[float],
    sample_b: List[float],
    margin: float
) -> Optional[Dict[str, Any]]:
    """
    Two One-Sided Tests (TOST) procedure testing statistical equivalence within bound (-margin, +margin).

    Args:
        sample_a: First observation sample.
        sample_b: Second observation sample.
        margin: Equivalence bound delta.

    Returns:
        Dictionary reporting observed difference, TOST p-value, and equivalence conclusion at alpha=0.05.
    """
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None

    diff = float(a.mean() - b.mean())
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se == 0:
        return {"diff": diff, "equivalent": abs(diff) < margin, "p_tost": float("nan")}

    t1 = (diff - (-margin)) / se
    t2 = (diff - margin) / se
    df = len(a) + len(b) - 2

    p1 = 1 - stats.t.cdf(t1, df)
    p2 = stats.t.cdf(t2, df)
    p_tost = max(p1, p2)

    return {
        "diff": round(diff, 4),
        "p_tost": round(float(p_tost), 6),
        "equivalent": bool(p_tost < 0.05)
    }


def build_methods(
    ranks: List[int],
    config: Dict[str, Any]
) -> List[Tuple[str, int, int, float, float, bool]]:
    """
    Construct experimental configurations grid (rank sweep and regularization variants).

    Returns:
        List of tuples: (method_label, rank, alpha, dropout, weight_decay, use_dora)
    """
    methods = []
    for r in ranks:
        alpha = alpha_for_rank(r, config["ALPHA_SCHEDULE"], config["ALPHA_FIXED_VALUE"])
        methods.append((f"LoRA_r{r}", r, alpha, 0.05, 0.0, False))

    if config.get("RUN_REGULARIZATION_CONDITIONS", True):
        mid_rank = ranks[len(ranks) // 2]
        a_mid = alpha_for_rank(mid_rank, config["ALPHA_SCHEDULE"], config["ALPHA_FIXED_VALUE"])
        methods.append((f"LoRA_r{mid_rank}_wd", mid_rank, a_mid, 0.05, config["WD_STRONG"], False))
        methods.append((f"LoRA_r{mid_rank}_drop", mid_rank, a_mid, config["DROPOUT_HIGH"], 0.0, False))
        methods.append((f"DoRA_r{mid_rank}", mid_rank, a_mid, 0.05, 0.0, True))

    return methods
