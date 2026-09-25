from fastapi.testclient import TestClient
from api.main import app

c = TestClient(app)

def test_health():
    assert c.get("/health").status_code == 200

def test_metrics():
    j = c.get("/metrics").json()
    assert "total_ead" in j

def test_counterparties():
    j = c.get("/counterparties").json()
    assert isinstance(j, list)
    if j:
        assert "mixed_nash_bank" in j[0]
        assert "value_of_perfect_info" in j[0]

def test_stress():
    j = c.get("/stress").json()
    assert "scenarios" in j
    assert "reverse_stress" in j

def test_transitions():
    j = c.get("/transitions").json()
    assert isinstance(j, dict)
