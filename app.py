"""
Tennis Match Prediction Tool
Flask app with offline fallback — works with or without API key.
"""

import json
import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request as req

from predictor import calculate_prediction
from backtester import run_backtest, run_full_backtest
from parlay import find_best_parlays
from nn_predict import nn_predict_event, load_nn_model

# Import mega predict (with fallback if not yet trained)
try:
    from mega_predict import predict_match as mega_predict_match, is_available as mega_available
except ImportError:
    mega_predict_match = None
    mega_available = lambda: False

# Import mega parlay
try:
    from mega_parlay import find_todays_parlays, backtest_parlays
except ImportError:
    find_todays_parlays = None
    backtest_parlays = None

# Import props predictor
try:
    from props_predictor import predict_slate as props_predict_slate
except ImportError:
    props_predict_slate = None

load_dotenv()

app = Flask(__name__)

API_BASE = "https://api.sportsgameodds.com/v2"
API_KEY = os.getenv("SGO_API_KEY")
HEADERS = {"X-Api-Key": API_KEY}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SNAPSHOT_PREDICTIONS = os.path.join(CACHE_DIR, "predictions_snapshot.json")
SNAPSHOT_UPCOMING = os.path.join(CACHE_DIR, "upcoming_snapshot.json")
SNAPSHOT_PARLAYS = os.path.join(CACHE_DIR, "parlays_snapshot.json")


def api_available():
    """Quick check if API key works."""
    if not API_KEY:
        return False
    try:
        resp = requests.get(
            f"{API_BASE}/sports", headers=HEADERS, params={"limit": 1}, timeout=5
        )
        return resp.status_code == 200
    except Exception:
        return False


# Check API on startup
_api_live = None


def is_api_live():
    global _api_live
    if _api_live is None:
        _api_live = api_available()
        if _api_live:
            print("[API] Key is active — live mode")
        else:
            print("[API] Key expired or missing — offline mode (using cached data)")
    return _api_live


def fetch_upcoming_events(league=None, limit=30):
    """Fetch upcoming tennis events. Falls back to cached snapshot."""
    if is_api_live():
        try:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            params = {
                "sportID": "TENNIS",
                "startsAfter": now,
                "limit": limit,
                "includeOdds": "true",
            }
            if league and league != "ALL":
                params["leagueID"] = league

            resp = requests.get(
                f"{API_BASE}/events", headers=HEADERS, params=params, timeout=15
            )
            resp.raise_for_status()
            events = resp.json().get("data", [])

            # Update snapshot
            with open(SNAPSHOT_UPCOMING, "w") as f:
                json.dump(events, f)

            return events
        except Exception:
            pass  # fall through to cache

    # Offline: use cached snapshot
    if os.path.exists(SNAPSHOT_UPCOMING):
        with open(SNAPSHOT_UPCOMING) as f:
            events = json.load(f)
        if league and league != "ALL":
            events = [e for e in events if e.get("leagueID") == league]
        return events[:limit]

    return []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predictions")
def get_predictions():
    """Return predictions. Falls back to cached snapshot if API is down."""
    league = req.args.get("league", "ALL")

    # Try live first
    try:
        events = fetch_upcoming_events(league=league)
        if events:
            predictions = []
            for event in events:
                pred = calculate_prediction(event)
                nn_pred = nn_predict_event(event)
                if nn_pred:
                    pred.update(nn_pred)
                # Mega ensemble prediction
                if mega_predict_match and mega_available():
                    try:
                        mega_pred = mega_predict_match(event)
                        if mega_pred:
                            pred.update(mega_pred)
                    except Exception:
                        pass
                predictions.append(pred)

            with_odds = [p for p in predictions if p["has_odds"]]
            without_odds = [p for p in predictions if not p["has_odds"]]
            with_odds.sort(key=lambda x: x.get("confidence", 0), reverse=True)

            result = {"success": True, "predictions": with_odds + without_odds}

            # Save snapshot
            with open(SNAPSHOT_PREDICTIONS, "w") as f:
                json.dump(result, f)

            offline = not is_api_live()
            result["offline"] = offline
            return jsonify(result)
    except Exception:
        pass

    # Fallback: cached predictions
    if os.path.exists(SNAPSHOT_PREDICTIONS):
        with open(SNAPSHOT_PREDICTIONS) as f:
            result = json.load(f)
        result["offline"] = True
        if league != "ALL":
            result["predictions"] = [
                p for p in result["predictions"] if p.get("league") == league
            ]
        return jsonify(result)

    return jsonify({"success": False, "error": "No API and no cached data", "offline": True}), 502


@app.route("/api/live")
def get_live():
    """Return live match data."""
    if not is_api_live():
        return jsonify({"success": True, "predictions": [], "offline": True})
    try:
        params = {
            "sportID": "TENNIS",
            "status": "live",
            "includeOdds": "true",
            "limit": 20,
        }
        resp = requests.get(f"{API_BASE}/events", headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        events = resp.json().get("data", [])
        predictions = [calculate_prediction(e) for e in events]
        return jsonify({"success": True, "predictions": predictions})
    except requests.RequestException as e:
        return jsonify({"success": False, "error": str(e)}), 502


@app.route("/api/backtest")
def get_backtest():
    """Run backtest — works fully offline using cached historical data."""
    period = req.args.get("period", "1month")

    now = datetime.now(timezone.utc)
    starts_before = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    if period == "all":
        try:
            results = run_full_backtest()
            return jsonify({"success": True, **results})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    period_map = {
        "1week": ("2026-03-10T00:00:00.000Z", 5),
        "2weeks": ("2026-03-03T00:00:00.000Z", 8),
        "1month": ("2026-02-17T00:00:00.000Z", 15),
        "3months": ("2025-12-17T00:00:00.000Z", 30),
        "6months": ("2025-09-17T00:00:00.000Z", 30),
        "1year": ("2025-03-17T00:00:00.000Z", 30),
        "2years": ("2024-03-17T00:00:00.000Z", 30),
    }

    starts_after, max_pages = period_map.get(period, period_map["1month"])

    try:
        results = run_backtest(starts_after, starts_before, max_pages=max_pages)
        return jsonify({"success": True, **results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/parlays")
def get_parlays():
    """Find best parlays. Falls back to cached snapshot."""
    bankroll = float(req.args.get("bankroll", 120))
    max_legs = int(req.args.get("max_legs", 4))
    league = req.args.get("league", "ALL")

    try:
        events = fetch_upcoming_events(league=league, limit=40)
        if events:
            predictions = [calculate_prediction(e) for e in events]
            with_odds = [p for p in predictions if p["has_odds"]]
            result = find_best_parlays(
                with_odds, max_legs=max_legs, bankroll=bankroll, top_n=15
            )
            full_result = {"success": True, **result}

            # Save snapshot
            with open(SNAPSHOT_PARLAYS, "w") as f:
                json.dump(full_result, f)

            return jsonify(full_result)
    except Exception:
        pass

    # Fallback
    if os.path.exists(SNAPSHOT_PARLAYS):
        with open(SNAPSHOT_PARLAYS) as f:
            result = json.load(f)
        result["offline"] = True
        return jsonify(result)

    return jsonify({"success": False, "error": "No API and no cached data"}), 502


@app.route("/api/status")
def get_status():
    """Return system status — API connectivity, cached data, model info."""
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.startswith("events_")]
    total_events = 0
    for cf in cache_files:
        with open(os.path.join(CACHE_DIR, cf)) as f:
            total_events += len(json.load(f))

    model_exists = os.path.exists(os.path.join(os.path.dirname(__file__), "model", "tennis_nn.pt"))
    mega_exists = mega_available()

    return jsonify({
        "api_live": is_api_live(),
        "cached_months": len(cache_files),
        "cached_events": total_events,
        "nn_model_loaded": model_exists,
        "mega_model_loaded": mega_exists,
        "has_predictions_snapshot": os.path.exists(SNAPSHOT_PREDICTIONS),
        "has_parlays_snapshot": os.path.exists(SNAPSHOT_PARLAYS),
    })


@app.route("/api/mega-parlays")
def get_mega_parlays():
    """Return mega-model-powered 3-leg parlays."""
    if find_todays_parlays is None:
        return jsonify({"success": False, "error": "Mega parlay not available"}), 501

    bankroll = float(req.args.get("bankroll", 120))
    min_conf = int(req.args.get("min_confidence", 60))
    num_legs = int(req.args.get("legs", 3))

    try:
        result = find_todays_parlays(
            num_legs=num_legs, min_confidence=min_conf, bankroll=bankroll
        )
        if result:
            return jsonify({"success": True, **result})
        return jsonify({"success": False, "error": "No data available"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/parlay-backtest")
def get_parlay_backtest():
    """Return cached parlay backtest results."""
    report_path = os.path.join(os.path.dirname(__file__), "model", "parlay_backtest.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            return jsonify({"success": True, **json.load(f)})
    return jsonify({"success": False, "error": "Run mega_parlay.py first"}), 404


@app.route("/api/props-parlay", methods=["POST"])
def get_props_parlay():
    """Predict Underdog Fantasy player props and return optimal parlay.

    Expects JSON body:
    {
        "matches": [
            {
                "player": "Iga Swiatek",
                "opponent": "Magda Linette",
                "surface": "Hard",
                "props": {"aces": 2.5, "total_games": 17.5, ...}
            },
            ...
        ]
    }
    """
    if props_predict_slate is None:
        return jsonify({"success": False, "error": "Props predictor not available"}), 501

    data = req.get_json(force=True, silent=True)
    if not data or "matches" not in data:
        return jsonify({"success": False, "error": "POST JSON with 'matches' array"}), 400

    try:
        results = props_predict_slate(data["matches"])
        return jsonify({"success": True, **results})
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/backtest")
def backtest_page():
    return render_template("backtest.html")


@app.route("/parlays")
def parlays_page():
    return render_template("parlays.html")


if __name__ == "__main__":
    app.run(debug=True, port=5050)
