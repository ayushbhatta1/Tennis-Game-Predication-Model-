import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtester import _american_profit, determine_actual_winner  # noqa: E402
from predictor import american_to_probability, find_value_bets  # noqa: E402


def test_american_to_probability():
    assert american_to_probability("-200") == 200 / 300
    assert american_to_probability("+150") == 100 / 250
    assert american_to_probability("EVEN") is None


def test_american_profit():
    assert _american_profit("+150") == 1.5
    assert _american_profit("-200") == 0.5
    assert _american_profit("EVEN") is None
    assert _american_profit(None) is None


def test_actual_winner_from_scores():
    ev = {"teams": {"home": {"score": 2}, "away": {"score": 1}}}
    assert determine_actual_winner(ev) == "home"
    assert determine_actual_winner({"teams": {}}) is None


def _odds(home, away):
    return {"home": {"by_bookmaker": {"bk": home}, "book_odds": home},
            "away": {"by_bookmaker": {"bk": away}, "book_odds": away}}


def test_value_bets_need_model_probability():
    pred = {"home": {"name": "A"}, "away": {"name": "B"}}
    assert find_value_bets(_odds("+120", "-140"), 0.45, 0.55, pred) == []
    pred["mega_home_prob"] = 70.0  # model 70% on a +120 (45%) home side
    bets = find_value_bets(_odds("+120", "-140"), 0.45, 0.55, pred)
    assert [b["side"] for b in bets] == ["home"]


def test_api_samples_skip_matches_already_in_store_history():
    import build_training_data as btd

    store = {"records": [{"date": "20250105"}, {"date": "20241230"}]}
    end = btd.store_last_date(store)
    assert end == "20250105"
    assert btd.stats_include_match(end, "20250105")      # same tournament week
    assert btd.stats_include_match(end, "20241201")      # inside history
    assert not btd.stats_include_match(end, "20250110")  # after history: fine
    assert not btd.stats_include_match("", "20250110")
