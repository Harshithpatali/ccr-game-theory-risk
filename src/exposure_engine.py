"""
Simplified SA-CCR style exposure calculation.

Replacement Cost (RC) = max(V - C, 0) for margined; max(V, 0) unmargined.
Potential Future Exposure (PFE) = multiplier * AddOn, with maturity scaling.
EAD = alpha * (RC + PFE), alpha = 1.4 (Basel).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ALPHA = 1.4


def calculate_exposure(t: pd.DataFrame) -> pd.DataFrame:
    x = t.copy()
    # Replacement cost at trade level
    x["rc_trade"] = np.where(
        x["margining"].eq("Margined"),
        np.maximum(x["mtm"] - x["collateral"], 0.0),
        np.maximum(x["mtm"], 0.0),
    )
    # Supervisory add-on with maturity factor sqrt(min(M,1) + (M-1)+ * indicator)
    # Simplified: notional * add_on_factor * min(1, sqrt(maturity))
    x["pfe_trade"] = x["notional"] * x["add_on_factor"] * np.minimum(1.0, np.sqrt(x["maturity_years"].clip(0.01)))

    g = (
        x.groupby(["legal_entity", "counterparty_id"], as_index=False)
        .agg(
            mtm=("mtm", "sum"),
            collateral=("collateral", "sum"),
            pfe=("pfe_trade", "sum"),
            rating=("rating", "first"),
            credit_limit=("credit_limit", "first"),
            trade_count=("trade_id", "count"),
            notional=("notional", "sum"),
        )
    )
    g["rc"] = np.maximum(g["mtm"] - g["collateral"], 0.0)
    # Multiplier (simplified SA-CCR floor at 0.05)
    g["multiplier"] = np.minimum(1.0, 0.05 + 0.95 * np.exp(g["rc"] / (2.0 * g["pfe"].clip(lower=1.0))))
    g["pfe_adj"] = g["multiplier"] * g["pfe"]
    g["ead"] = ALPHA * (g["rc"] + g["pfe_adj"])
    g["limit_utilisation"] = g["ead"] / g["credit_limit"].clip(lower=1.0)
    g["breach"] = g["limit_utilisation"] >= 1.0
    return g.sort_values("ead", ascending=False)
