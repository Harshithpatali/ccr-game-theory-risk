from src.data_generator import generate_portfolio
from src.statistical_engine import add_features
from src.probability_engine import fit, predict
from src.monte_carlo import breach_probability

def test_risk():
    t, d = generate_portfolio(200, 30)
    x = add_features(d)
    assert len(x) == len(d)
    assert "anomaly_score" in x
    assert "ewma_vol" in x
    assert "mahalanobis" in x
    m = fit(x)
    row = x.iloc[-1].to_dict()
    p = predict(m, row)
    assert abs(sum(p.values()) - 1.0) < 1e-6
    mc = breach_probability(1e7, 2e7, vol=0.2, n=500, days=10)
    assert 0 <= mc["breach_probability"] <= 1
