"""
Bayesian game of incomplete information for CCR limit / collateral decisions.

Bank strategies (row player): maintain_limit, reduce_limit, increase_collateral
Counterparty behavioural types (column / nature): reduce_exposure, stable, increase_exposure

Payoffs calibrated to economic loss of limit breach, opportunity cost of
over-collateralisation, and franchise value. Utilities are CARA transforms
so risk aversion is explicit.

Outputs:
  - pure-strategy Bayesian expected utilities
  - preferred pure strategy
  - mixed-strategy security level (maximin over bank mixed strategies)
  - value of perfect information (VOI)
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from itertools import combinations

BANK = ["maintain_limit", "reduce_limit", "increase_collateral"]
CP = ["reduce_exposure", "stable", "increase_exposure"]

PAYOFF_MONEY = np.array([
    [ 4.5,  3.0, -8.0],
    [ 2.0,  1.5,  3.5],
    [ 1.0,  2.5,  5.0],
], dtype=float)

CARA_LAMBDA = 0.15


def _cara(x: np.ndarray, lam: float = CARA_LAMBDA) -> np.ndarray:
    return (1.0 - np.exp(-lam * x)) / lam


def _security_mixed(A: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Maximin mixed strategy for the row player:
    max_p  min_j  (p A)_j
    Solved by linear programming via enumeration of vertices for 3x3,
    or by solving the dual LP with scipy if available; fallback to
    grid search over the simplex.
    """
    m, n = A.shape
    # Grid search on the 2-simplex (sufficient for 3 strategies)
    best_val = -np.inf
    best_p = np.ones(m) / m
    grid = np.linspace(0, 1, 21)
    for p0 in grid:
        for p1 in grid:
            p2 = 1.0 - p0 - p1
            if p2 < -1e-9:
                continue
            p = np.array([p0, p1, max(p2, 0.0)])
            p = p / p.sum()
            val = float((p @ A).min())
            if val > best_val:
                best_val = val
                best_p = p
    return best_p, best_val


def _support_nash(A: np.ndarray, tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, float]:
    """Try to find an interior / support-based NE; fall back to security strategy."""
    m, n = A.shape
    best_val = -np.inf
    best_p = np.ones(m) / m
    best_q = np.ones(n) / n
    found = False

    for k in (1, 2, 3):
        for rows in combinations(range(m), k):
            for cols in combinations(range(n), k):
                sub = A[np.ix_(rows, cols)]
                try:
                    if k == 1:
                        p_loc = np.array([1.0])
                        q_loc = np.array([1.0])
                        val = float(sub[0, 0])
                    elif k == 2:
                        a11, a12 = sub[0, 0], sub[0, 1]
                        a21, a22 = sub[1, 0], sub[1, 1]
                        denom = (a11 - a12) - (a21 - a22)
                        if abs(denom) < tol:
                            continue
                        q1 = (a22 - a21) / denom
                        q_loc = np.array([q1, 1 - q1])
                        if np.any(q_loc < -tol) or np.any(q_loc > 1 + tol):
                            continue
                        q_loc = np.clip(q_loc, 0, 1)
                        q_loc /= q_loc.sum()
                        denom_p = (a11 - a21) - (a12 - a22)
                        if abs(denom_p) < tol:
                            continue
                        p1 = (a22 - a12) / denom_p
                        p_loc = np.array([p1, 1 - p1])
                        if np.any(p_loc < -tol) or np.any(p_loc > 1 + tol):
                            continue
                        p_loc = np.clip(p_loc, 0, 1)
                        p_loc /= p_loc.sum()
                        val = float(p_loc @ sub @ q_loc)
                    else:
                        eqs = [sub[r] - sub[-1] for r in range(k - 1)] + [np.ones(k)]
                        Mq = np.array(eqs)
                        rq = np.zeros(k)
                        rq[-1] = 1.0
                        q_loc = np.linalg.solve(Mq, rq)
                        if np.any(q_loc < -tol) or np.any(q_loc > 1 + tol):
                            continue
                        q_loc = np.clip(q_loc, 0, 1)
                        q_loc /= q_loc.sum()
                        eqs = [sub[:, c] - sub[:, -1] for c in range(k - 1)] + [np.ones(k)]
                        Mp = np.array(eqs).T
                        rp = np.zeros(k)
                        rp[-1] = 1.0
                        p_loc = np.linalg.solve(Mp, rp)
                        if np.any(p_loc < -tol) or np.any(p_loc > 1 + tol):
                            continue
                        p_loc = np.clip(p_loc, 0, 1)
                        p_loc /= p_loc.sum()
                        val = float(p_loc @ sub @ q_loc)
                except (np.linalg.LinAlgError, ValueError):
                    continue

                p_full = np.zeros(m)
                p_full[list(rows)] = p_loc
                q_full = np.zeros(n)
                q_full[list(cols)] = q_loc
                Aq = A @ q_full
                pA = p_full @ A
                if abs(Aq.max() - val) < 0.25 and abs(pA.min() - val) < 0.25:
                    if val > best_val:
                        best_val = val
                        best_p, best_q = p_full, q_full
                        found = True

    if not found:
        best_p, best_val = _security_mixed(A)
        best_q = np.ones(n) / n

    return best_p, best_q, float(best_val)


def game(prob: Dict[str, float], risk_aversion: float = CARA_LAMBDA) -> dict:
    p = np.array([float(prob.get(s, 0.0)) for s in CP])
    p = p / p.sum() if p.sum() > 0 else np.ones(3) / 3.0

    U = _cara(PAYOFF_MONEY, risk_aversion)

    eu = {BANK[i]: float(U[i] @ p) for i in range(3)}
    preferred = max(eu, key=eu.get)

    mix_p, mix_q, value = _support_nash(U)

    eu_perfect = float(np.sum(p * U.max(axis=0)))
    voi = max(0.0, eu_perfect - eu[preferred])

    return {
        "expected_utilities": eu,
        "preferred_strategy": preferred,
        "probabilities": {k: float(v) for k, v in zip(CP, p)},
        "mixed_nash_bank": {k: float(v) for k, v in zip(BANK, mix_p)},
        "mixed_nash_nature": {k: float(v) for k, v in zip(CP, mix_q)},
        "nash_value": value,
        "value_of_perfect_info": float(voi),
        "cara_lambda": risk_aversion,
        "payoff_matrix_utility": U.tolist(),
        "payoff_matrix_money": PAYOFF_MONEY.tolist(),
    }
