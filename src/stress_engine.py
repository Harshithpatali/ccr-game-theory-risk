"""
Multi-factor stress testing with linear sensitivities and nonlinear multipliers.

Scenarios combine instantaneous shocks to rates, FX, equity and credit spreads
with a residual systemic multiplier.  Incremental EAD and breach counts are
reported; a simple reverse-stress utilisation threshold is also computed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Instantaneous relative shock to EAD under each named scenario
SCENARIOS = {
    "rates_up_200bp": {"rate": 0.020, "fx": 0.0, "equity": 0.0, "credit": 0.005, "systemic": 1.05},
    "fx_shock_10pct": {"rate": 0.0, "fx": 0.10, "equity": 0.0, "credit": 0.01, "systemic": 1.08},
    "equity_selloff_20": {"rate": 0.005, "fx": 0.02, "equity": -0.20, "credit": 0.03, "systemic": 1.12},
    "credit_widening_100bp": {"rate": 0.0, "fx": 0.0, "equity": -0.05, "credit": 0.10, "systemic": 1.15},
    "combined_severe": {"rate": 0.015, "fx": 0.08, "equity": -0.15, "credit": 0.08, "systemic": 1.35},
}

# Linear beta of EAD to each risk factor (illustrative, portfolio-level)
BETAS = {"rate": 0.8, "fx": 1.1, "equity": 0.6, "credit": 1.4}


def _stressed_ead(ead: np.ndarray, shocks: dict) -> np.ndarray:
    rel = 0.0
    for f, beta in BETAS.items():
        rel += beta * shocks.get(f, 0.0)
    return ead * (1.0 + rel) * shocks.get("systemic", 1.0)


def summary(d: pd.DataFrame) -> pd.DataFrame:
    base = float(d["ead"].sum())
    limit = d["credit_limit"].values
    ead = d["ead"].values
    rows = []
    for name, shocks in SCENARIOS.items():
        se = _stressed_ead(ead, shocks)
        rows.append({
            "scenario": name,
            "stressed_ead": float(se.sum()),
            "incremental_ead": float(se.sum() - base),
            "breaches": int((se > limit).sum()),
            "breach_rate": float((se > limit).mean()),
            "max_utilisation": float((se / np.maximum(limit, 1.0)).max()),
        })
    return pd.DataFrame(rows)


def reverse_stress_utilisation(d: pd.DataFrame, target_breach_rate: float = 0.25) -> dict:
    """
    Find the uniform multiplicative shock to EAD that produces a given
    portfolio breach rate (simple reverse stress).
    """
    ead = d["ead"].values
    limit = d["credit_limit"].values
    # binary search on multiplier
    lo, hi = 1.0, 5.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        rate = (ead * mid > limit).mean()
        if rate < target_breach_rate:
            lo = mid
        else:
            hi = mid
    return {
        "target_breach_rate": target_breach_rate,
        "ead_multiplier": float(0.5 * (lo + hi)),
        "implied_incremental_ead": float(ead.sum() * (0.5 * (lo + hi) - 1.0)),
    }
