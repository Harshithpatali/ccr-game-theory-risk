import pandas as pd

def scenario_multipliers():
    return {"rates_up":1.12,"fx_shock":1.15,"equity_selloff":1.22,"credit_widening":1.35,"combined":1.60}

def summary(d):
    latest=d.sort_values("date").groupby("counterparty_id").tail(1).copy(); rows=[]
    for name,m in scenario_multipliers().items():
        stressed=latest.ead*m
        rows.append({"scenario":name,"baseline_ead":float(latest.ead.sum()),"stressed_ead":float(stressed.sum()),
                     "incremental_ead":float((stressed-latest.ead).sum()),
                     "breached_counterparties":int((stressed>latest.credit_limit).sum()),
                     "max_utilisation":float((stressed/latest.credit_limit).max())})
    return pd.DataFrame(rows)
