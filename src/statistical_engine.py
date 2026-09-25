"""
Statistical feature engineering and anomaly detection for CCR time series.

Features:
  - Rolling percentage change and EWMA volatility of EAD
  - Collateral coverage ratio
  - Multivariate Mahalanobis distance (robust covariance)
  - Isolation Forest + Local Outlier Factor ensemble score
  - Autocorrelation of utilisation (persistence signal)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import MinCovDet


FEATURE_COLS = [
    "limit_utilisation",
    "ead_change_pct",
    "market_volatility",
    "credit_spread_change",
    "collateral_coverage",
    "ewma_vol",
    "util_autocorr",
]


def _ewma_vol(series: pd.Series, span: int = 10) -> pd.Series:
    """Exponentially weighted moving standard deviation."""
    return series.ewm(span=span, min_periods=3).std().fillna(0.0)


def _rolling_autocorr(series: pd.Series, window: int = 15, lag: int = 1) -> pd.Series:
    """Rolling lag-1 autocorrelation of utilisation (persistence)."""
    def ac(x):
        if len(x) < lag + 2:
            return 0.0
        a, b = x[:-lag], x[lag:]
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    return series.rolling(window, min_periods=5).apply(ac, raw=True).fillna(0.0)


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    x = d.sort_values(["counterparty_id", "date"]).copy()

    # Core derived series
    x["ead_change_pct"] = (
        x.groupby("counterparty_id")["ead"]
        .pct_change()
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )
    x["collateral_coverage"] = (x["collateral"] / x["ead"].clip(lower=1.0)).clip(0, 1.5)
    x["ewma_vol"] = x.groupby("counterparty_id")["ead_change_pct"].transform(_ewma_vol)
    x["util_autocorr"] = x.groupby("counterparty_id")["limit_utilisation"].transform(
        _rolling_autocorr
    )

    cols = [c for c in FEATURE_COLS if c in x.columns]
    X = x[cols].fillna(0.0).astype(float)

    # Isolation Forest
    iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42, n_jobs=1)
    iso.fit(X)
    iso_score = -iso.decision_function(X)  # higher = more anomalous
    iso_flag = iso.predict(X) == -1

    # Local Outlier Factor (novelty-style score)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.03, novelty=False, n_jobs=1)
    lof_pred = lof.fit_predict(X)
    lof_score = -lof.negative_outlier_factor_  # higher = more anomalous
    lof_flag = lof_pred == -1

    # Robust Mahalanobis distance (MinCovDet)
    try:
        mcd = MinCovDet(random_state=42).fit(X)
        mahal = mcd.mahalanobis(X)
        # chi-square 99% threshold for df = n_features
        from scipy.stats import chi2
        thresh = chi2.ppf(0.99, df=X.shape[1])
        mahal_flag = mahal > thresh
        # normalise to [0,1]-ish
        mahal_score = mahal / (thresh + 1e-9)
    except Exception:
        mahal = np.zeros(len(X))
        mahal_score = np.zeros(len(X))
        mahal_flag = np.zeros(len(X), dtype=bool)

    # Ensemble: average of standardised scores
    def _z(s):
        s = np.asarray(s, dtype=float)
        mu, sd = s.mean(), s.std()
        return (s - mu) / (sd + 1e-9)

    ensemble = (_z(iso_score) + _z(lof_score) + _z(mahal_score)) / 3.0
    # map to roughly [0,1] via logistic
    anomaly_score = 1.0 / (1.0 + np.exp(-ensemble))

    x["anomaly_score"] = anomaly_score
    x["anomaly_flag"] = anomaly_score > 0.72
    x["iso_score"] = iso_score
    x["lof_score"] = lof_score
    x["mahalanobis"] = mahal
    return x
