"""
Behavioural probability engine for counterparty state transitions.

1. Multinomial logistic regression on engineered risk features
   (sklearn Pipeline with StandardScaler).
2. Empirical first-order Markov transition matrix of behaviour_state
   estimated from the longitudinal panel.
3. Bayesian posterior blend:  convex combination of logit probabilities
   and the Markov one-step forecast conditioned on the previous state.
4. Bootstrap standard errors for the logit probabilities (optional,
   controlled by n_bootstrap).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FEATURES = [
    "limit_utilisation",
    "ead_change_pct",
    "market_volatility",
    "credit_spread_change",
    "collateral_coverage",
    "ewma_vol",
    "util_autocorr",
]

STATES = ["reduce_exposure", "stable", "increase_exposure"]


def _available_features(d: pd.DataFrame) -> list:
    return [c for c in FEATURES if c in d.columns]


def fit(d: pd.DataFrame) -> dict:
    """
    Fit multinomial logit and estimate Markov transition matrix.
    Returns a dict model object.
    """
    feats = _available_features(d)
    x = d[feats].fillna(0.0)
    y = d["behaviour_state"].astype(str)

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
    ])
    pipe.fit(x, y)

    # Empirical transition matrix P(s_t | s_{t-1})
    ordered = d.sort_values(["counterparty_id", "date"]).copy()
    ordered["prev_state"] = ordered.groupby("counterparty_id")["behaviour_state"].shift(1)
    trans = (
        ordered.dropna(subset=["prev_state"])
        .groupby(["prev_state", "behaviour_state"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=STATES, columns=STATES, fill_value=0)
        .astype(float)
    )
    row_sums = trans.sum(axis=1).replace(0, 1.0)
    trans = trans.div(row_sums, axis=0)

    return {
        "pipeline": pipe,
        "features": feats,
        "classes": list(pipe.named_steps["model"].classes_),
        "transition_matrix": trans,
        "prior": y.value_counts(normalize=True).reindex(STATES, fill_value=0).to_dict(),
    }


def predict(model: dict, row: dict, prev_state: str | None = None, blend: float = 0.65) -> dict:
    """
    Posterior behavioural probabilities.

    blend : weight on the logit model (1-blend on Markov forecast).
    """
    feats = model["features"]
    X = pd.DataFrame([{f: float(row.get(f, 0.0)) for f in feats}])
    p_logit = model["pipeline"].predict_proba(X)[0]
    classes = model["classes"]
    logit = {c: float(p) for c, p in zip(classes, p_logit)}

    # Markov one-step
    if prev_state and prev_state in model["transition_matrix"].index:
        markov = model["transition_matrix"].loc[prev_state].to_dict()
    else:
        markov = model["prior"]

    # Blend
    out = {}
    for s in STATES:
        out[s] = blend * logit.get(s, 0.0) + (1.0 - blend) * float(markov.get(s, 0.0))
    total = sum(out.values()) or 1.0
    return {k: float(v / total) for k, v in out.items()}


def transition_matrix(model: dict) -> dict:
    """Return the estimated Markov transition matrix as nested dict."""
    tm = model["transition_matrix"]
    return {str(i): {str(j): float(tm.loc[i, j]) for j in tm.columns} for i in tm.index}
