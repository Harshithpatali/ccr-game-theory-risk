from src.game_engine import game

def test_game():
    r = game({"reduce_exposure": 0.2, "stable": 0.5, "increase_exposure": 0.3})
    assert r["preferred_strategy"] in r["expected_utilities"]
    assert "mixed_nash_bank" in r
    assert "value_of_perfect_info" in r
    assert abs(sum(r["probabilities"].values()) - 1.0) < 1e-6
