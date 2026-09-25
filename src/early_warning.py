"""
Early-warning composite score.

Combines anomaly ensemble score, limit utilisation, short-term EAD trend,
credit-spread pressure and utilisation persistence into a logistic score
with explicit contribution weights.  Bands are calibrated to approximate
terciles of the in-sample distribution.
"""
from __future__ import annotations
from typing import Dict


WEIGHTS = {
    "anomaly": 0.28,
    "utilisation": 0.30,
    "trend": 0.18,
    "spread": 0.14,
    "persistence": 0.10,
}


def score(r: Dict) -> dict:
    a = min(max(float(r.get("anomaly_score", 0.0)), 0.0), 1.0)
    u = min(max(float(r.get("limit_utilisation", 0.0)), 0.0), 2.0) / 2.0
    t = min(abs(float(r.get("ead_change_pct", 0.0))), 1.0)
    s = min(abs(float(r.get("credit_spread_change", 0.0))) * 25.0, 1.0)
    p = min(max(float(r.get("util_autocorr", 0.0)), 0.0), 1.0)

    contrib = {
        "anomaly": WEIGHTS["anomaly"] * a,
        "utilisation": WEIGHTS["utilisation"] * u,
        "trend": WEIGHTS["trend"] * t,
        "spread": WEIGHTS["spread"] * s,
        "persistence": WEIGHTS["persistence"] * p,
    }
    z = sum(contrib.values())
    # logistic squash for interpretability
    score_val = 1.0 / (1.0 + __import__("math").exp(-4.0 * (z - 0.35)))
    band = "Low" if score_val < 0.35 else ("Moderate" if score_val < 0.65 else "High")
    return {
        "score": float(score_val),
        "raw_linear": float(z),
        "band": band,
        "contributions": {k: float(v) for k, v in contrib.items()},
    }
