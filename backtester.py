"""
Backtest engine for Tennis Match Prediction Model

Fetches completed matches with odds going back to 2024,
runs predictions retroactively, compares to actual results,
and generates comprehensive performance metrics.
"""

import os
import json
import time
from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta

import requests
from dotenv import load_dotenv

from predictor import calculate_prediction, american_to_probability

load_dotenv()

API_BASE = "https://api.sportsgameodds.com/v2"
API_KEY = os.getenv("SGO_API_KEY")
HEADERS = {"X-Api-Key": API_KEY}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def determine_actual_winner(event):
    """Determine who actually won from results data."""
    results = event.get("results", {})
    teams = event.get("teams", {})

    home_score = teams.get("home", {}).get("score")
    away_score = teams.get("away", {}).get("score")

    if home_score is not None and away_score is not None:
        if home_score > away_score:
            return "home"
        elif away_score > home_score:
            return "away"

    game_results = results.get("game", {}) or results.get("reg", {})
    if game_results:
        home_pts = game_results.get("home", {}).get("points", 0)
        away_pts = game_results.get("away", {}).get("points", 0)
        if home_pts > away_pts:
            return "home"
        elif away_pts > home_pts:
            return "away"

    return None


def fetch_month_events(year, month, max_pages=60):
    """Fetch all completed tennis events with odds for a given month."""
    cache_file = os.path.join(CACHE_DIR, f"events_{year}_{month:02d}.json")

    now = datetime.now(timezone.utc)
    is_current_month = (year == now.year and month == now.month)

    if os.path.exists(cache_file):
        if is_current_month:
            # Current month: re-fetch if cache is older than 1 hour
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < 3600:
                with open(cache_file) as f:
                    return json.load(f)
        else:
            # Past months: cache is permanent
            with open(cache_file) as f:
                return json.load(f)

    starts_after = f"{year}-{month:02d}-01T00:00:00.000Z"
    if month == 12:
        starts_before = f"{year + 1}-01-01T00:00:00.000Z"
    else:
        starts_before = f"{year}-{month + 1:02d}-01T00:00:00.000Z"

    all_events = []
    cursor = None
    page = 0

    while page < max_pages:
        params = {
            "sportID": "TENNIS",
            "startsAfter": starts_after,
            "startsBefore": starts_before,
            "includeOdds": "true",
            "limit": 50,
        }
        if cursor:
            params["cursor"] = cursor

        try:
            resp = requests.get(
                f"{API_BASE}/events", headers=HEADERS, params=params, timeout=20
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            break

        events = data.get("data", [])
        if not events:
            break

        for e in events:
            status = e.get("status", {})
            if (
                status.get("completed")
                and not status.get("cancelled")
                and e.get("odds")
                and e.get("results")
            ):
                all_events.append(e)

        cursor = data.get("nextCursor")
        if not cursor:
            break

        page += 1
        time.sleep(0.25)

    # Cache to disk
    with open(cache_file, "w") as f:
        json.dump(all_events, f)

    return all_events


def fetch_all_events(start_year=2024, start_month=2):
    """Fetch all available historical data month by month."""
    now = datetime.now(timezone.utc)
    all_events = []

    year, month = start_year, start_month
    while (year < now.year) or (year == now.year and month <= now.month):
        events = fetch_month_events(year, month)
        all_events.extend(events)

        month += 1
        if month > 12:
            month = 1
            year += 1

    return all_events


def analyze_events(events):
    """Run predictions on all events and compute stats."""
    # Try loading NN model for comparison
    try:
        from nn_predict import nn_predict_event, load_nn_model
        nn_model, nn_mean, nn_std = load_nn_model()
        has_nn = nn_model is not None
    except Exception:
        has_nn = False

    # Try loading mega model
    try:
        from mega_predict import predict_match as mega_predict_match, is_available as mega_available
        has_mega = mega_available()
    except Exception:
        has_mega = False
        mega_predict_match = None

    # Mega stats
    mega_total = 0
    mega_correct = 0
    mega_calibration = {}
    for bucket in ["50-60", "60-70", "70-80", "80-90", "90-100"]:
        mega_calibration[bucket] = {"predicted": 0, "correct": 0, "total": 0}
    mega_monthly = {}

    match_results = []
    correct = 0
    total = 0
    total_with_opening = 0
    correct_with_opening = 0

    # NN stats
    nn_total = 0
    nn_correct = 0
    nn_opening_total = 0
    nn_opening_correct = 0
    nn_calibration = {}
    for bucket in ["50-60", "60-70", "70-80", "80-90", "90-100"]:
        nn_calibration[bucket] = {"predicted": 0, "correct": 0, "total": 0}
    nn_monthly = {}

    calibration = {}
    for bucket in ["50-60", "60-70", "70-80", "80-90", "90-100"]:
        calibration[bucket] = {"predicted": 0, "correct": 0, "total": 0}

    conf_buckets = {
        "high": {"correct": 0, "total": 0},
        "medium": {"correct": 0, "total": 0},
        "low": {"correct": 0, "total": 0},
    }

    # Yearly breakdown
    yearly_stats = {}

    # Monthly accuracy for chart
    monthly_stats = {}

    # League breakdown
    league_stats = {"ATP": {"correct": 0, "total": 0}, "WTA": {"correct": 0, "total": 0}}

    upsets = []
    value_bet_results = []
    value_profit = 0.0
    value_wins = 0

    for event in events:
        # Determine if opening odds are available
        odds_obj = event.get("odds", {})
        hml = odds_obj.get("points-home-game-ml-home", {})
        has_opening = bool(hml.get("openFairOdds"))

        prediction = calculate_prediction(event, use_opening=has_opening)
        if not prediction["has_odds"]:
            continue

        actual_winner = determine_actual_winner(event)
        if actual_winner is None:
            continue

        predicted_winner = prediction.get("predicted_winner")
        winner_prob = prediction.get("winner_prob", 50)
        confidence = prediction.get("confidence", 0)
        is_correct = predicted_winner == actual_winner

        total += 1
        if is_correct:
            correct += 1

        if has_opening:
            total_with_opening += 1
            if is_correct:
                correct_with_opening += 1

        # NN prediction
        nn_pred = None
        nn_correct_flag = None
        if has_nn:
            nn_pred = nn_predict_event(event)
            if nn_pred:
                nn_winner = nn_pred["nn_winner"]
                nn_is_correct = nn_winner == actual_winner
                nn_total += 1
                if nn_is_correct:
                    nn_correct += 1
                nn_correct_flag = nn_is_correct

                if has_opening:
                    nn_opening_total += 1
                    if nn_is_correct:
                        nn_opening_correct += 1

                # NN calibration
                nn_wp = max(nn_pred["nn_home_prob"], nn_pred["nn_away_prob"])
                if nn_wp < 60:
                    nb = "50-60"
                elif nn_wp < 70:
                    nb = "60-70"
                elif nn_wp < 80:
                    nb = "70-80"
                elif nn_wp < 90:
                    nb = "80-90"
                else:
                    nb = "90-100"
                nn_calibration[nb]["predicted"] += nn_wp
                nn_calibration[nb]["total"] += 1
                if nn_is_correct:
                    nn_calibration[nb]["correct"] += 1

                # NN monthly
                sa = event.get("status", {}).get("startsAt", "")
                mm = sa[:7] if sa else "????"
                if mm not in nn_monthly:
                    nn_monthly[mm] = {"correct": 0, "total": 0}
                nn_monthly[mm]["total"] += 1
                if nn_is_correct:
                    nn_monthly[mm]["correct"] += 1

        # Mega prediction
        mega_pred = None
        mega_correct_flag = None
        if has_mega and mega_predict_match:
            try:
                mega_pred = mega_predict_match(event)
            except Exception:
                pass
            if mega_pred:
                mega_winner = mega_pred["mega_winner"]
                mega_is_correct = mega_winner == actual_winner
                mega_total += 1
                if mega_is_correct:
                    mega_correct += 1
                mega_correct_flag = mega_is_correct

                # Mega calibration
                mega_wp = max(mega_pred["mega_home_prob"], mega_pred["mega_away_prob"])
                if mega_wp < 60:
                    mb = "50-60"
                elif mega_wp < 70:
                    mb = "60-70"
                elif mega_wp < 80:
                    mb = "70-80"
                elif mega_wp < 90:
                    mb = "80-90"
                else:
                    mb = "90-100"
                mega_calibration[mb]["predicted"] += mega_wp
                mega_calibration[mb]["total"] += 1
                if mega_is_correct:
                    mega_calibration[mb]["correct"] += 1

                # Mega monthly
                sa = event.get("status", {}).get("startsAt", "")
                mm = sa[:7] if sa else "????"
                if mm not in mega_monthly:
                    mega_monthly[mm] = {"correct": 0, "total": 0}
                mega_monthly[mm]["total"] += 1
                if mega_is_correct:
                    mega_monthly[mm]["correct"] += 1

        # Actual winner info
        if actual_winner == "home":
            actual_name = prediction["home"]["name"]
        else:
            actual_name = prediction["away"]["name"]

        starts_at = prediction.get("starts_at", "")
        match_date = starts_at[:10] if starts_at else ""
        match_year = match_date[:4] if match_date else "????"
        match_month = match_date[:7] if match_date else "????"
        league = prediction.get("league", "???")

        result = {
            "home": prediction["home"]["name"],
            "away": prediction["away"]["name"],
            "league": league,
            "date": starts_at,
            "predicted_winner": prediction.get("winner_name", "?"),
            "predicted_prob": winner_prob,
            "actual_winner": actual_name,
            "correct": is_correct,
            "confidence": confidence,
            "upset": not is_correct and winner_prob >= 60,
            "has_opening_odds": has_opening,
            "nn_correct": nn_correct_flag,
            "nn_prob": max(nn_pred["nn_home_prob"], nn_pred["nn_away_prob"]) if nn_pred else None,
            "mega_correct": mega_correct_flag,
            "mega_prob": max(mega_pred["mega_home_prob"], mega_pred["mega_away_prob"]) if mega_pred else None,
        }

        # --- Yearly ---
        if match_year not in yearly_stats:
            yearly_stats[match_year] = {"correct": 0, "total": 0}
        yearly_stats[match_year]["total"] += 1
        if is_correct:
            yearly_stats[match_year]["correct"] += 1

        # --- Monthly ---
        if match_month not in monthly_stats:
            monthly_stats[match_month] = {"correct": 0, "total": 0}
        monthly_stats[match_month]["total"] += 1
        if is_correct:
            monthly_stats[match_month]["correct"] += 1

        # --- League ---
        if league in league_stats:
            league_stats[league]["total"] += 1
            if is_correct:
                league_stats[league]["correct"] += 1

        # --- Calibration ---
        if winner_prob < 60:
            bucket = "50-60"
        elif winner_prob < 70:
            bucket = "60-70"
        elif winner_prob < 80:
            bucket = "70-80"
        elif winner_prob < 90:
            bucket = "80-90"
        else:
            bucket = "90-100"

        calibration[bucket]["predicted"] += winner_prob
        calibration[bucket]["total"] += 1
        if is_correct:
            calibration[bucket]["correct"] += 1

        # --- Confidence ---
        if confidence >= 70:
            conf_key = "high"
        elif confidence >= 40:
            conf_key = "medium"
        else:
            conf_key = "low"
        conf_buckets[conf_key]["total"] += 1
        if is_correct:
            conf_buckets[conf_key]["correct"] += 1

        # --- Value Bets ---
        for vb in prediction.get("value_bets", []):
            vb_won = vb["side"] == actual_winner
            odds_val = int(vb["book_odds"])
            if vb_won:
                profit = odds_val / 100 if odds_val > 0 else 100 / abs(odds_val)
            else:
                profit = -1.0
            value_bet_results.append({
                "match": f"{prediction['home']['name']} vs {prediction['away']['name']}",
                "bookmaker": vb["bookmaker"],
                "player": vb["player"],
                "odds": vb["book_odds"],
                "edge": vb["edge"],
                "won": vb_won,
                "profit": round(profit, 2),
                "date": match_date,
            })
            value_profit += profit
            if vb_won:
                value_wins += 1

        if result["upset"]:
            upsets.append(result)

        match_results.append(result)

    # --- Compute final stats ---
    for bucket in calibration:
        b = calibration[bucket]
        if b["total"] > 0:
            b["actual_pct"] = round(b["correct"] / b["total"] * 100, 1)
            b["avg_predicted_pct"] = round(b["predicted"] / b["total"], 1)
        else:
            b["actual_pct"] = 0
            b["avg_predicted_pct"] = 0

    for key in conf_buckets:
        b = conf_buckets[key]
        b["accuracy"] = round(b["correct"] / b["total"] * 100, 1) if b["total"] > 0 else 0

    for y in yearly_stats:
        ys = yearly_stats[y]
        ys["accuracy"] = round(ys["correct"] / ys["total"] * 100, 1) if ys["total"] > 0 else 0

    monthly_accuracy = []
    for m in sorted(monthly_stats.keys()):
        ms = monthly_stats[m]
        acc = round(ms["correct"] / ms["total"] * 100, 1) if ms["total"] > 0 else 0
        monthly_accuracy.append({"month": m, "accuracy": acc, "total": ms["total"]})

    for lg in league_stats:
        ls = league_stats[lg]
        ls["accuracy"] = round(ls["correct"] / ls["total"] * 100, 1) if ls["total"] > 0 else 0

    accuracy = round(correct / total * 100, 1) if total > 0 else 0
    total_vb = len(value_bet_results)
    opening_acc = round(correct_with_opening / total_with_opening * 100, 1) if total_with_opening > 0 else 0

    # Get date range
    dates = [m["date"][:10] for m in match_results if m.get("date")]
    date_range = f"{min(dates)} to {max(dates)}" if dates else "N/A"

    return {
        "summary": {
            "total_matches": total,
            "correct": correct,
            "accuracy": accuracy,
            "date_range": date_range,
            "total_value_bets": total_vb,
            "value_wins": value_wins,
            "value_win_rate": round(value_wins / total_vb * 100, 1) if total_vb > 0 else 0,
            "value_profit_units": round(value_profit, 2),
            "value_roi": round(value_profit / total_vb * 100, 1) if total_vb > 0 else 0,
            "num_upsets": len(upsets),
            "matches_with_opening_odds": total_with_opening,
            "opening_odds_accuracy": opening_acc,
        },
        "calibration": calibration,
        "confidence_accuracy": conf_buckets,
        "yearly_stats": yearly_stats,
        "monthly_accuracy": monthly_accuracy,
        "league_stats": league_stats,
        "matches": match_results,
        "value_bets": value_bet_results[:50],
        "upsets": upsets[:30],
        "nn_stats": _build_nn_stats(
            nn_total, nn_correct, nn_opening_total, nn_opening_correct,
            nn_calibration, nn_monthly, has_nn
        ),
        "mega_stats": _build_model_stats(
            mega_total, mega_correct, mega_calibration, mega_monthly, has_mega
        ),
    }


def _build_model_stats(total, correct, calibration, monthly, available):
    """Build generic model comparison stats."""
    if not available or total == 0:
        return {"available": False}

    for bucket in calibration:
        b = calibration[bucket]
        if b["total"] > 0:
            b["actual_pct"] = round(b["correct"] / b["total"] * 100, 1)
            b["avg_predicted_pct"] = round(b["predicted"] / b["total"], 1)
        else:
            b["actual_pct"] = 0
            b["avg_predicted_pct"] = 0

    monthly_list = []
    for m in sorted(monthly.keys()):
        ms = monthly[m]
        acc = round(ms["correct"] / ms["total"] * 100, 1) if ms["total"] > 0 else 0
        monthly_list.append({"month": m, "accuracy": acc, "total": ms["total"]})

    return {
        "available": True,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
        "calibration": calibration,
        "monthly": monthly_list,
    }


def _build_nn_stats(nn_total, nn_correct, nn_opening_total, nn_opening_correct,
                     nn_calibration, nn_monthly, has_nn):
    """Build NN comparison stats."""
    if not has_nn or nn_total == 0:
        return {"available": False}

    # Calibration
    for bucket in nn_calibration:
        b = nn_calibration[bucket]
        if b["total"] > 0:
            b["actual_pct"] = round(b["correct"] / b["total"] * 100, 1)
            b["avg_predicted_pct"] = round(b["predicted"] / b["total"], 1)
        else:
            b["actual_pct"] = 0
            b["avg_predicted_pct"] = 0

    # Monthly
    nn_monthly_list = []
    for m in sorted(nn_monthly.keys()):
        ms = nn_monthly[m]
        acc = round(ms["correct"] / ms["total"] * 100, 1) if ms["total"] > 0 else 0
        nn_monthly_list.append({"month": m, "accuracy": acc, "total": ms["total"]})

    return {
        "available": True,
        "total": nn_total,
        "correct": nn_correct,
        "accuracy": round(nn_correct / nn_total * 100, 1) if nn_total > 0 else 0,
        "opening_total": nn_opening_total,
        "opening_correct": nn_opening_correct,
        "opening_accuracy": round(nn_opening_correct / nn_opening_total * 100, 1) if nn_opening_total > 0 else 0,
        "calibration": nn_calibration,
        "monthly": nn_monthly_list,
    }


def run_backtest(starts_after, starts_before, max_pages=10, use_cache=True):
    """
    Run backtest over a date range.
    For "5year" / "all" periods, fetches all available data.
    """
    # Parse the date range to determine months
    sa_dt = datetime.fromisoformat(starts_after.replace("Z", "+00:00"))
    sb_dt = datetime.fromisoformat(starts_before.replace("Z", "+00:00"))

    all_events = []
    year, month = sa_dt.year, sa_dt.month

    while (year < sb_dt.year) or (year == sb_dt.year and month <= sb_dt.month):
        events = fetch_month_events(year, month, max_pages=max_pages)
        # Filter to the requested date range
        for e in events:
            sa = e.get("status", {}).get("startsAt", "")
            if sa >= starts_after and sa <= starts_before:
                all_events.append(e)

        month += 1
        if month > 12:
            month = 1
            year += 1

    return analyze_events(all_events)


def run_full_backtest():
    """Run backtest over ALL available data (2024-present)."""
    all_events = fetch_all_events(start_year=2024, start_month=2)
    return analyze_events(all_events)


if __name__ == "__main__":
    results = run_full_backtest()
    s = results["summary"]
    print(f"\n=== FULL BACKTEST ===")
    print(f"Period: {s['date_range']}")
    print(f"Matches: {s['total_matches']}")
    print(f"Accuracy: {s['accuracy']}% ({s['correct']}/{s['total_matches']})")
    print(f"Opening odds matches: {s['matches_with_opening_odds']} ({s['opening_odds_accuracy']}% acc)")
    print(f"\nYearly:")
    for y, ys in results["yearly_stats"].items():
        print(f"  {y}: {ys['accuracy']}% ({ys['correct']}/{ys['total']})")
    print(f"\nCalibration:")
    for bucket, data in results["calibration"].items():
        if data["total"] > 0:
            print(f"  {bucket}%: pred {data['avg_predicted_pct']}% → actual {data['actual_pct']}% ({data['total']})")
