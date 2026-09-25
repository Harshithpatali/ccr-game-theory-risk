"""
Multi-factor Monte Carlo engine for limit-breach probability.

Exposure path is modelled as a geometric Brownian motion with
stochastic volatility driven by an Ornstein-Uhlenbeck process
(mean-reverting log-vol).  Optional wrong-way risk correlation
between credit-spread shocks and exposure.

Outputs:
  - P(max_t EAD_t > limit) over a horizon of H business days
  - quantiles of the maximum exposure distribution
  - expected shortfall of the peak exposure
"""
from __future__ import annotations
import numpy as np


def breach_probability(
    ead: float,
    limit: float,
    vol: float = 0.20,
    n: int = 8000,
    days: int = 30,
    kappa: float = 2.0,
    theta: float = None,
    xi: float = 0.30,
    rho_wwr: float = 0.25,
    seed: int = 42,
) -> dict:
    """
    Simulate peak exposure under GBM + OU stochastic volatility.

    Parameters
    ----------
    ead, limit : current exposure and credit limit
    vol : initial annualised volatility
    n, days : paths and horizon
    kappa, theta, xi : OU speed, long-run mean (log-vol), vol-of-vol
    rho_wwr : correlation between brownian drivers (wrong-way risk)
    """
    if theta is None:
        theta = np.log(max(vol, 0.05))

    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    sqrt_dt = np.sqrt(dt)

    # Correlated Brownian increments: Z1 for price, Z2 for vol
    Z = rng.standard_normal((n, days, 2))
    Z[:, :, 1] = rho_wwr * Z[:, :, 0] + np.sqrt(max(1 - rho_wwr**2, 0)) * Z[:, :, 1]

    log_vol = np.full(n, np.log(max(vol, 0.01)))
    log_ead = np.full(n, np.log(max(ead, 1.0)))
    peak = np.full(n, ead)

    for t in range(days):
        # OU step for log-vol
        log_vol = log_vol + kappa * (theta - log_vol) * dt + xi * sqrt_dt * Z[:, t, 1]
        sigma_t = np.exp(log_vol)
        # GBM step for exposure
        log_ead = log_ead - 0.5 * sigma_t**2 * dt + sigma_t * sqrt_dt * Z[:, t, 0]
        ead_t = np.exp(log_ead)
        peak = np.maximum(peak, ead_t)

    breach = (peak > limit).mean()
    return {
        "breach_probability": float(breach),
        "p95_max_ead": float(np.quantile(peak, 0.95)),
        "p99_max_ead": float(np.quantile(peak, 0.99)),
        "expected_shortfall_99": float(peak[peak >= np.quantile(peak, 0.99)].mean()) if breach > 0 else float(np.quantile(peak, 0.99)),
        "mean_peak_ead": float(peak.mean()),
        "horizon_days": days,
        "n_paths": n,
        "wwr_rho": rho_wwr,
    }
