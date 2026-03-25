"""
Mega Parlay Engine — uses the mega ensemble to build and backtest 3-leg parlays.

Strategy:
  1. Score each match using mega models (XGBoost + Deep LSTM + ensemble)
  2. Pick the top N most confident predictions per day
  3. Build 3-leg parlays from those picks
  4. Use best available bookmaker odds for each leg
  5. Backtest across all historical data (2024-2026)

Backtest outputs:
  - Win rate, ROI, profit/loss by month
  - Comparison: mega picks vs random picks vs favorites-only
  - Optimal confidence thresholds
"""

import json
import os
import glob
import pickle
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations

import numpy as np

from predictor import american_to_probability, calculate_prediction
from nn_model import determine_winner

BASE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, "cache")
MODEL_DIR = os.path.join(BASE_DIR, "model")


def odds_to_decimal(american_odds_str):
    try:
        odds = int(american_odds_str)
    except (ValueError, TypeError):
        return None
    if odds > 0:
        return 1 + odds / 100
    elif odds < 0:
        return 1 + 100 / abs(odds)
    return 2.0


def decimal_to_american(dec):
    if dec >= 2.0:
        return f"+{int((dec - 1) * 100)}"
    elif dec > 1.0:
        return f"{int(-100 / (dec - 1))}"
    return "+100"


def load_xgb_model():
    """Load the mega XGBoost model for fast backtesting."""
    path = os.path.join(MODEL_DIR, "mega_xgb.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def xgb_predict_event(event, xgb_models):
    """Fast XGBoost prediction for a single event."""
    from feature_engine import extract_odds_features_from_event, build_feature_vector, NUM_FEATURES
    from player_resolver import load_mapping, get_sackmann_id

    if xgb_models is None:
        return None

    model = xgb_models.get("full")
    if model is None:
        return None

    # Extract odds features
    hml = event.get("odds", {}).get("points-home-game-ml-home", {})
    has_opening = bool(hml.get("openFairOdds"))
    odds_features = extract_odds_features_from_event(event, use_opening=has_opening)
    if odds_features is None:
        return None

    league = event.get("leagueID", "ATP")
    vec = build_feature_vector(
        odds_features=odds_features,
        home_stats={},
        away_stats={},
        league=league,
        has_odds=True,
        has_history=False,
    )
    vec = np.nan_to_num(vec, 0.0).reshape(1, -1)

    prob = model.predict_proba(vec)[0, 1]
    return {
        "home_prob": prob,
        "away_prob": 1 - prob,
        "winner": "home" if prob > 0.5 else "away",
        "confidence": abs(prob - 0.5) * 200,
        "winner_prob": max(prob, 1 - prob) * 100,
    }


def get_best_book_odds(event, side):
    """Find the best (highest payout) bookmaker odds for a side."""
    odds = event.get("odds", {})
    key = f"points-{side}-game-ml-{side}"
    market = odds.get(key, {})

    # Start with consensus
    best_odds_str = market.get("openBookOdds") or market.get("bookOdds")
    best_decimal = odds_to_decimal(best_odds_str) if best_odds_str else None
    best_book = "consensus"

    # Check individual bookmakers for better odds
    for bk_name, bk_data in market.get("byBookmaker", {}).items():
        bk_odds = bk_data.get("openOdds") or bk_data.get("odds")
        if bk_odds:
            dec = odds_to_decimal(bk_odds)
            if dec and (best_decimal is None or dec > best_decimal):
                best_decimal = dec
                best_odds_str = bk_odds
                best_book = bk_name

    return best_odds_str, best_decimal, best_book


def score_match_for_parlay(event, xgb_models):
    """Score a match and return parlay leg info if viable."""
    pred = xgb_predict_event(event, xgb_models)
    if pred is None:
        return None

    teams = event.get("teams", {})
    home_name = teams.get("home", {}).get("names", {}).get("long", "?")
    away_name = teams.get("away", {}).get("names", {}).get("long", "?")

    winner_side = pred["winner"]
    winner_name = home_name if winner_side == "home" else away_name
    loser_name = away_name if winner_side == "home" else home_name

    # Get best bookmaker odds for the predicted winner
    best_odds_str, best_decimal, best_book = get_best_book_odds(event, winner_side)

    if best_decimal is None or best_decimal <= 1.0:
        return None

    implied_prob = 1 / best_decimal if best_decimal > 0 else 0.5

    return {
        "event_id": event.get("eventID", ""),
        "home": home_name,
        "away": away_name,
        "pick": winner_name,
        "pick_side": winner_side,
        "opponent": loser_name,
        "league": event.get("leagueID", "?"),
        "date": event.get("status", {}).get("startsAt", "")[:10],
        "model_prob": pred["winner_prob"],
        "confidence": pred["confidence"],
        "book_odds": best_odds_str,
        "decimal_odds": round(best_decimal, 3),
        "implied_prob": round(implied_prob * 100, 1),
        "best_book": best_book,
        "event": event,
    }


def _make_parlay(legs):
    """Build a single parlay dict from a list of legs."""
    parlay_decimal = 1.0
    true_prob = 1.0
    implied_prob = 1.0
    for leg in legs:
        parlay_decimal *= leg["decimal_odds"]
        true_prob *= leg["model_prob"] / 100
        implied_prob *= leg["implied_prob"] / 100

    ev = (true_prob * parlay_decimal) - 1
    avg_prob = sum(l["model_prob"] for l in legs) / len(legs)
    min_prob = min(l["model_prob"] for l in legs)

    return {
        "legs": legs,
        "num_legs": len(legs),
        "parlay_odds": decimal_to_american(parlay_decimal),
        "decimal_odds": round(parlay_decimal, 3),
        "true_prob": round(true_prob * 100, 2),
        "implied_prob": round(implied_prob * 100, 2),
        "ev_per_dollar": round(ev, 4),
        "ev_pct": round(ev * 100, 1),
        "avg_leg_prob": round(avg_prob, 1),
        "min_leg_prob": round(min_prob, 1),
    }


def build_daily_parlays(day_legs, num_legs=3, min_confidence=60, top_picks=8):
    """Build N-leg parlays from a single day's matches."""
    viable = [l for l in day_legs if l["confidence"] >= min_confidence]
    viable.sort(key=lambda x: x["model_prob"], reverse=True)
    viable = viable[:top_picks]

    if len(viable) < num_legs:
        return []

    # For large parlays (7+ legs), skip combinatorics — just build a few
    # smart parlays from the sorted list instead of brute-forcing all combos
    if num_legs >= 7:
        parlays = []
        # #1: Top N by model probability (safest)
        parlays.append(_make_parlay(viable[:num_legs]))

        # #2-5: Swap out the weakest leg(s) with alternatives
        if len(viable) > num_legs:
            for swap_idx in range(min(num_legs, 4)):
                alt_legs = viable[:num_legs].copy()
                # Replace leg at swap_idx with the next available
                for alt in viable[num_legs:]:
                    alt_legs_try = [l for i, l in enumerate(alt_legs) if i != swap_idx] + [alt]
                    parlays.append(_make_parlay(alt_legs_try))
                    break

        # #6+: Slide the window (legs 1-N, 2-N+1, etc.)
        for offset in range(1, min(len(viable) - num_legs + 1, 5)):
            parlays.append(_make_parlay(viable[offset:offset + num_legs]))

        parlays.sort(key=lambda x: x["true_prob"] * 0.6 + x["ev_per_dollar"] * 40, reverse=True)
        # Deduplicate by leg set
        seen = set()
        unique = []
        for p in parlays:
            key = frozenset(l["event_id"] for l in p["legs"])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    # For small parlays (2-6 legs), use full combinatorics
    parlays = []
    for combo in combinations(viable, num_legs):
        parlays.append(_make_parlay(list(combo)))

    parlays.sort(key=lambda x: x["true_prob"] * 0.6 + x["ev_per_dollar"] * 40, reverse=True)
    return parlays


def check_parlay_result(parlay, results_map):
    """Check if a parlay won by looking up actual match results."""
    all_won = True
    leg_results = []

    for leg in parlay["legs"]:
        eid = leg["event_id"]
        event = leg["event"]

        actual = determine_winner(event)
        if actual is None:
            return None, []  # Can't determine

        actual_winner = "home" if actual == 1 else "away"
        leg_won = actual_winner == leg["pick_side"]

        leg_results.append({
            "pick": leg["pick"],
            "opponent": leg["opponent"],
            "won": leg_won,
            "model_prob": leg["model_prob"],
            "book_odds": leg["book_odds"],
        })

        if not leg_won:
            all_won = False

    return all_won, leg_results


def backtest_parlays(
    min_confidence=60,
    num_legs=3,
    top_picks=8,
    picks_per_day=1,
    bet_size=10.0,
    start_date="2024-06-01",
    end_date="2026-03-18",
):
    """
    Backtest the 3-leg parlay strategy across all historical data.

    Strategy: each day, pick the best 3-leg parlay from top-confidence matches.
    """
    print("Loading models...")
    xgb_models = load_xgb_model()
    if xgb_models is None:
        print("ERROR: No XGBoost model found. Run mega_xgb_trainer.py first.")
        return None

    print("Loading events...")
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "events_*.json")))
    all_events = []
    for filepath in files:
        with open(filepath) as f:
            events = json.load(f)
        for e in events:
            s = e.get("status", {})
            if s.get("completed") and not s.get("cancelled") and e.get("odds"):
                date = s.get("startsAt", "")[:10]
                if start_date <= date <= end_date:
                    all_events.append(e)

    print(f"  {len(all_events)} completed events in range")

    # Group events by date
    by_date = defaultdict(list)
    for e in all_events:
        date = e["status"]["startsAt"][:10]
        by_date[date].append(e)

    dates = sorted(by_date.keys())
    print(f"  {len(dates)} days with matches")

    # Score all matches
    print("Scoring matches...")
    scored_by_date = {}
    total_scored = 0

    for date in dates:
        day_events = by_date[date]
        day_legs = []
        for event in day_events:
            leg = score_match_for_parlay(event, xgb_models)
            if leg:
                day_legs.append(leg)
                total_scored += 1
        scored_by_date[date] = day_legs

    print(f"  {total_scored} matches scored across {len(dates)} days")

    # Build and evaluate parlays day by day
    print(f"\nBacktesting (min_confidence={min_confidence}, {num_legs}-leg, "
          f"top_picks={top_picks}, ${bet_size}/bet)...\n")

    total_bets = 0
    total_won = 0
    total_wagered = 0.0
    total_payout = 0.0
    monthly_results = defaultdict(lambda: {
        "bets": 0, "won": 0, "wagered": 0.0, "payout": 0.0, "legs_correct": 0, "legs_total": 0
    })
    all_parlay_results = []
    confidence_buckets = defaultdict(lambda: {"bets": 0, "won": 0})

    for date in dates:
        day_legs = scored_by_date[date]
        if len(day_legs) < num_legs:
            continue

        # Build parlays for this day
        parlays = build_daily_parlays(
            day_legs, num_legs=num_legs,
            min_confidence=min_confidence, top_picks=top_picks
        )

        if not parlays:
            continue

        # Take top N parlays per day
        for parlay in parlays[:picks_per_day]:
            won, leg_results = check_parlay_result(parlay, {})
            if won is None:
                continue

            month = date[:7]
            total_bets += 1
            total_wagered += bet_size

            payout = bet_size * parlay["decimal_odds"] if won else 0
            total_payout += payout

            if won:
                total_won += 1

            monthly_results[month]["bets"] += 1
            monthly_results[month]["wagered"] += bet_size
            monthly_results[month]["payout"] += payout
            if won:
                monthly_results[month]["won"] += 1

            # Track individual legs
            for lr in leg_results:
                monthly_results[month]["legs_total"] += 1
                if lr["won"]:
                    monthly_results[month]["legs_correct"] += 1

            # Confidence bucket
            bucket = int(parlay["avg_leg_prob"] / 5) * 5
            confidence_buckets[bucket]["bets"] += 1
            if won:
                confidence_buckets[bucket]["won"] += 1

            all_parlay_results.append({
                "date": date,
                "won": won,
                "parlay_odds": parlay["parlay_odds"],
                "decimal_odds": parlay["decimal_odds"],
                "true_prob": parlay["true_prob"],
                "avg_leg_prob": parlay["avg_leg_prob"],
                "min_leg_prob": parlay["min_leg_prob"],
                "bet": bet_size,
                "payout": round(payout, 2),
                "legs": leg_results,
            })

    # Results
    profit = total_payout - total_wagered
    roi = (profit / total_wagered * 100) if total_wagered > 0 else 0
    win_rate = (total_won / total_bets * 100) if total_bets > 0 else 0

    # Leg-level accuracy
    all_legs_correct = sum(
        sum(1 for lr in pr["legs"] if lr["won"])
        for pr in all_parlay_results
    )
    all_legs_total = sum(len(pr["legs"]) for pr in all_parlay_results)
    leg_accuracy = (all_legs_correct / all_legs_total * 100) if all_legs_total > 0 else 0

    print(f"{'='*60}")
    print(f"  3-LEG PARLAY BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Period:       {start_date} to {end_date}")
    print(f"  Strategy:     Top {num_legs}-leg parlay per day (min {min_confidence}% conf)")
    print(f"  Bet size:     ${bet_size:.2f}")
    print(f"")
    print(f"  Total bets:   {total_bets}")
    print(f"  Won:          {total_won} ({win_rate:.1f}%)")
    print(f"  Wagered:      ${total_wagered:.2f}")
    print(f"  Payouts:      ${total_payout:.2f}")
    print(f"  Profit:       ${profit:+.2f}")
    print(f"  ROI:          {roi:+.1f}%")
    print(f"")
    print(f"  Leg accuracy: {all_legs_correct}/{all_legs_total} ({leg_accuracy:.1f}%)")
    print(f"  Avg odds:     {np.mean([p['decimal_odds'] for p in all_parlay_results]):.2f}x" if all_parlay_results else "")

    # Monthly breakdown
    print(f"\n  Monthly Breakdown:")
    print(f"  {'Month':10s} {'Bets':>5s} {'Won':>5s} {'Rate':>6s} {'Wagered':>9s} {'Payout':>9s} {'P/L':>9s} {'ROI':>7s} {'Legs':>7s}")
    for month in sorted(monthly_results.keys()):
        mr = monthly_results[month]
        m_profit = mr["payout"] - mr["wagered"]
        m_roi = (m_profit / mr["wagered"] * 100) if mr["wagered"] > 0 else 0
        m_rate = (mr["won"] / mr["bets"] * 100) if mr["bets"] > 0 else 0
        m_leg_acc = (mr["legs_correct"] / mr["legs_total"] * 100) if mr["legs_total"] > 0 else 0
        print(f"  {month:10s} {mr['bets']:5d} {mr['won']:5d} {m_rate:5.1f}% "
              f"${mr['wagered']:8.2f} ${mr['payout']:8.2f} ${m_profit:+8.2f} {m_roi:+6.1f}% "
              f"{m_leg_acc:5.1f}%")

    # Confidence analysis
    print(f"\n  Win Rate by Avg Leg Confidence:")
    for bucket in sorted(confidence_buckets.keys()):
        cb = confidence_buckets[bucket]
        rate = (cb["won"] / cb["bets"] * 100) if cb["bets"] > 0 else 0
        print(f"    {bucket}-{bucket+5}%: {cb['won']}/{cb['bets']} ({rate:.1f}%)")

    # Best/worst streaks
    if all_parlay_results:
        streak = 0
        best_streak = 0
        worst_streak = 0
        cold = 0
        for pr in all_parlay_results:
            if pr["won"]:
                streak += 1
                cold = 0
                best_streak = max(best_streak, streak)
            else:
                streak = 0
                cold += 1
                worst_streak = max(worst_streak, cold)

        avg_payout_on_win = (
            np.mean([p["payout"] for p in all_parlay_results if p["won"]])
            if total_won > 0 else 0
        )

        print(f"\n  Best win streak:  {best_streak}")
        print(f"  Worst loss streak: {worst_streak}")
        print(f"  Avg payout (wins): ${avg_payout_on_win:.2f}")

    # Save results
    results = {
        "config": {
            "min_confidence": min_confidence,
            "num_legs": num_legs,
            "top_picks": top_picks,
            "picks_per_day": picks_per_day,
            "bet_size": bet_size,
            "start_date": start_date,
            "end_date": end_date,
        },
        "summary": {
            "total_bets": total_bets,
            "total_won": total_won,
            "win_rate": round(win_rate, 1),
            "total_wagered": round(total_wagered, 2),
            "total_payout": round(total_payout, 2),
            "profit": round(profit, 2),
            "roi": round(roi, 1),
            "leg_accuracy": round(leg_accuracy, 1),
        },
        "monthly": {
            m: {
                "bets": mr["bets"],
                "won": mr["won"],
                "wagered": round(mr["wagered"], 2),
                "payout": round(mr["payout"], 2),
                "profit": round(mr["payout"] - mr["wagered"], 2),
            }
            for m, mr in sorted(monthly_results.items())
        },
        "parlays": all_parlay_results[-50:],  # Last 50 for review
    }

    out_path = os.path.join(MODEL_DIR, "parlay_backtest.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return results


def find_todays_parlays(num_legs=3, min_confidence=60, top_picks=8, bankroll=120):
    """Find the best 3-leg parlays for today's upcoming matches."""
    xgb_models = load_xgb_model()
    if xgb_models is None:
        return None

    # Load upcoming snapshot
    snapshot = os.path.join(CACHE_DIR, "upcoming_snapshot.json")
    if not os.path.exists(snapshot):
        return None

    with open(snapshot) as f:
        events = json.load(f)

    # Score all upcoming matches
    legs = []
    for event in events:
        leg = score_match_for_parlay(event, xgb_models)
        if leg:
            legs.append(leg)

    if len(legs) < num_legs:
        return {"parlays": [], "legs": legs, "message": f"Only {len(legs)} viable legs found"}

    # Build parlays
    parlays = build_daily_parlays(legs, num_legs=num_legs,
                                   min_confidence=min_confidence, top_picks=top_picks)

    # Add bet sizing
    for p in parlays:
        max_pct = 0.02
        ev_factor = max(0.1, min(1.0, (p["ev_per_dollar"] + 0.15) / 0.15))
        bet = round(bankroll * max_pct * ev_factor, 2)
        bet = max(1, min(bet, bankroll * 0.05))
        p["suggested_bet"] = bet
        p["potential_payout"] = round(bet * p["decimal_odds"], 2)
        p["potential_profit"] = round(p["potential_payout"] - bet, 2)

    return {
        "parlays": parlays[:10],
        "legs": legs,
        "total_viable": len(legs),
        "total_combos": len(parlays),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "today":
        result = find_todays_parlays()
        if result and result["parlays"]:
            print(f"\nTop 3-Leg Parlays for Today:")
            print(f"{'='*60}")
            for i, p in enumerate(result["parlays"][:5], 1):
                print(f"\n  Parlay #{i} — {p['parlay_odds']} ({p['decimal_odds']:.2f}x)")
                print(f"  Win prob: {p['true_prob']:.1f}% | EV: {p['ev_pct']:+.1f}%")
                for leg in p["legs"]:
                    print(f"    {leg['pick']:25s} ({leg['book_odds']:>6s}) "
                          f"vs {leg['opponent']:25s} — {leg['model_prob']:.1f}%")
        else:
            print("No parlays available")
    else:
        # Run full backtest
        print("Running parlay backtest with multiple configs...\n")

        # Main strategy
        backtest_parlays(min_confidence=60, num_legs=3, picks_per_day=1, bet_size=10)

        print("\n\n" + "="*60)
        print("  HIGHER CONFIDENCE THRESHOLD")
        print("="*60)
        backtest_parlays(min_confidence=80, num_legs=3, picks_per_day=1, bet_size=10)

        print("\n\n" + "="*60)
        print("  LOWER CONFIDENCE (MORE BETS)")
        print("="*60)
        backtest_parlays(min_confidence=40, num_legs=3, picks_per_day=1, bet_size=10)
