from fastapi import FastAPI, HTTPException
from src.data_generator import load_data
from src.statistical_engine import add_features
from src.probability_engine import fit, predict, transition_matrix
from src.game_engine import game
from src.monte_carlo import breach_probability
from src.stress_engine import summary, reverse_stress_utilisation
from src.early_warning import score

app = FastAPI(
    title="CCR Mathematical Risk API",
    version="2.0",
    description="Rigorous quantitative CCR layer: Bayesian games, Markov behaviour, multi-factor Monte Carlo, ensemble anomaly detection.",
)

_cache = {}


def prep(force: bool = False):
    if not force and "data" in _cache:
        return _cache["data"]
    t, d = load_data()
    d = add_features(d)
    model = fit(d)
    _cache["data"] = (t, d, model)
    return t, d, model


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0"}


@app.get("/metrics")
def metrics():
    t, d, _ = prep()
    return {
        "trades": len(t),
        "counterparties": int(d.counterparty_id.nunique()),
        "total_ead": float(d.ead.sum()),
        "anomalies": int(d.anomaly_flag.sum()),
        "mean_utilisation": float(d.limit_utilisation.mean()),
        "days": int(d.date.nunique()),
    }


@app.get("/transitions")
def transitions():
    _, _, m = prep()
    return transition_matrix(m)


@app.get("/counterparties")
def cps():
    _, d, m = prep()
    latest = d.sort_values("date").groupby("counterparty_id").tail(1)
    # previous state for Markov
    ordered = d.sort_values(["counterparty_id", "date"])
    prev = ordered.groupby("counterparty_id")["behaviour_state"].shift(1)
    ordered = ordered.assign(prev_state=prev)
    prev_map = ordered.groupby("counterparty_id").tail(1).set_index("counterparty_id")["prev_state"].to_dict()

    out = []
    for _, r in latest.iterrows():
        row = r.to_dict()
        prev_s = prev_map.get(r.counterparty_id)
        prob = predict(m, row, prev_state=prev_s)
        g = game(prob)
        vol = max(0.05, float(r.market_volatility) * 0.18)
        mc = breach_probability(float(r.ead), float(r.credit_limit), vol=vol)
        ew = score(row)
        out.append({
            "counterparty_id": r.counterparty_id,
            "ead": float(r.ead),
            "pfe": float(r.pfe),
            "limit": float(r.credit_limit),
            "limit_utilisation": float(r.limit_utilisation),
            "anomaly": bool(r.anomaly_flag),
            "anomaly_score": float(r.anomaly_score),
            "early_warning_score": ew["score"],
            "early_warning_band": ew["band"],
            "early_warning_contributions": ew["contributions"],
            "breach_probability": mc["breach_probability"],
            "p95_max_ead": mc["p95_max_ead"],
            "p99_max_ead": mc["p99_max_ead"],
            "expected_shortfall_99": mc["expected_shortfall_99"],
            "preferred_strategy": g["preferred_strategy"],
            "probabilities": prob,
            "expected_utilities": g["expected_utilities"],
            "mixed_nash_bank": g["mixed_nash_bank"],
            "nash_value": g["nash_value"],
            "value_of_perfect_info": g["value_of_perfect_info"],
            "ewma_vol": float(r.get("ewma_vol", 0) or 0),
            "util_autocorr": float(r.get("util_autocorr", 0) or 0),
            "mahalanobis": float(r.get("mahalanobis", 0) or 0),
        })
    return out


@app.get("/counterparty/{cp}")
def cp(cp: str):
    rows = [x for x in cps() if x["counterparty_id"] == cp]
    if not rows:
        raise HTTPException(404, "Counterparty not found")
    return rows[0]


@app.get("/stress")
def stress():
    _, d, _ = prep()
    s = summary(d).to_dict("records")
    rev = reverse_stress_utilisation(d)
    return {"scenarios": s, "reverse_stress": rev}
