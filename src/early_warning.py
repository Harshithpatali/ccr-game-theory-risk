import numpy as np

def score(row):
    anomaly=np.clip(float(row.get("anomaly_score",0))/0.35,0,1)
    util=np.clip(float(row.get("limit_utilisation",0)),0,1.5)/1.5
    trend=np.clip(abs(float(row.get("ead_change_pct",0)))/.20,0,1)
    spread=np.clip(abs(float(row.get("credit_spread_change",0)))/.02,0,1)
    s=.30*anomaly+.30*util+.20*trend+.20*spread
    band="Low" if s<.35 else "Moderate" if s<.65 else "High"
    return {"score":float(s),"band":band}
