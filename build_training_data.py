"""
Build Training Data — combine historical + API matches into training arrays.

- Historical matches (2000-2024): full history features, has_odds=0, odds zeroed
- API matches (2024-2026): full features with odds, resolved via player_resolver
- Temporal split: train < 2026-03-01, test >= 2026-03-01
- Random home/away assignment with deterministic hash
"""

import json
import os
import pickle
import hashlib

import numpy as np

from feature_engine import (
    build_feature_vector, flip_features, extract_odds_features_from_event,
    NUM_FEATURES, FEATURE_NAMES
)
from player_resolver import load_mapping, get_sackmann_id
from build_feature_store import FeatureStoreBuilder, load_players, load_rankings
from nn_model import determine_winner

BASE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, "cache")
MODEL_DIR = os.path.join(BASE_DIR, "model")
STORE_FILE = os.path.join(MODEL_DIR, "feature_store.pkl")


def deterministic_flip(key_str):
    """Deterministic coin flip based on hash of a string."""
    h = hashlib.md5(key_str.encode()).hexdigest()
    return int(h, 16) % 2 == 0


def build_historical_samples(store):
    """Build training samples from historical match records."""
    records = store["records"]
    print(f"  Processing {len(records)} historical match records...")

    X, y, meta = [], [], []
    skipped = 0

    for record in records:
        date = record["date"]
        # Only use matches from 2000+
        if date < "20000101":
            continue

        w_stats = record["winner_stats"]
        l_stats = record["loser_stats"]
        surface = record["surface"]
        round_val = record["round"]
        level = record["tourney_level"]
        best_of = record["best_of"]
        league = record["league"]

        # Build feature vector (winner = home perspective)
        vec = build_feature_vector(
            odds_features=None,
            home_stats=w_stats,
            away_stats=l_stats,
            surface=surface,
            round_val=round_val,
            tourney_level=level,
            best_of=best_of,
            league=league,
            has_odds=False,
            has_history=True,
        )

        if np.isnan(vec).any():
            skipped += 1
            continue

        # Deterministic home/away assignment
        match_key = f"{record['tourney_id']}_{record['match_num']}"
        if deterministic_flip(match_key):
            # Winner is "home"
            X.append(vec)
            y.append(1)
        else:
            # Winner is "away" — flip features
            X.append(flip_features(vec))
            y.append(0)

        meta.append({
            "date": date,
            "league": league,
            "surface": surface,
            "source": "historical",
            "winner": record.get("winner_name", ""),
            "loser": record.get("loser_name", ""),
        })

    print(f"  Historical: {len(X)} samples ({skipped} skipped)")
    return X, y, meta


def build_api_samples(store):
    """Build training samples from API cached events."""
    import glob as globmod

    mapping = load_mapping()
    files = sorted(globmod.glob(os.path.join(CACHE_DIR, "events_*.json")))

    print(f"  Processing {len(files)} API cache files...")

    # Rebuild a live FeatureStoreBuilder from the stored state
    # We need it to look up player stats for API players via their Sackmann IDs
    players = store["players"]
    rankings = store["rankings"]

    X, y, meta = [], [], []
    matched = 0
    unmatched = 0
    no_winner = 0

    for filepath in files:
        with open(filepath) as f:
            events = json.load(f)

        for event in events:
            status = event.get("status", {})
            if not status.get("completed") or status.get("cancelled"):
                continue

            winner = determine_winner(event)
            if winner is None:
                no_winner += 1
                continue

            # Get team IDs
            home_tid = event["teams"]["home"].get("teamID", "")
            away_tid = event["teams"]["away"].get("teamID", "")
            if not home_tid or not away_tid:
                continue

            # Resolve to Sackmann IDs
            home_sid = get_sackmann_id(home_tid, mapping)
            away_sid = get_sackmann_id(away_tid, mapping)

            # Extract odds
            hml = event.get("odds", {}).get("points-home-game-ml-home", {})
            has_opening = bool(hml.get("openFairOdds"))
            odds_features = extract_odds_features_from_event(
                event, use_opening=has_opening
            )

            # Get match date
            date = status.get("startsAt", "")[:10]
            date_compact = date.replace("-", "")

            # Build player stats from store
            league = event.get("leagueID", "ATP")
            has_history = bool(home_sid) and bool(away_sid)

            if has_history:
                # Look up stats from the store's rolling state
                # Get stats for both players
                home_stats = _lookup_player_stats(
                    store, home_sid, away_sid, date_compact, "Hard"
                )
                away_stats = _lookup_player_stats(
                    store, away_sid, home_sid, date_compact, "Hard"
                )
                matched += 1
            else:
                home_stats = {}
                away_stats = {}
                unmatched += 1

            # Build feature vector (home perspective)
            vec = build_feature_vector(
                odds_features=odds_features,
                home_stats=home_stats,
                away_stats=away_stats,
                surface="Hard",  # API doesn't always expose surface
                round_val="R32",
                tourney_level="A",
                best_of=3,
                league=league,
                has_odds=odds_features is not None,
                has_history=has_history,
            )

            if np.isnan(vec).any():
                continue

            # Label: winner == 1 means home won
            label = winner  # 1 = home, 0 = away

            X.append(vec)
            y.append(label)
            meta.append({
                "date": date_compact,
                "league": league,
                "surface": "Hard",
                "source": "api",
                "has_opening": has_opening,
                "home": event["teams"]["home"]["names"]["long"],
                "away": event["teams"]["away"]["names"]["long"],
                "event_id": event.get("eventID", ""),
            })

    print(f"  API: {len(X)} samples (matched={matched}, unmatched={unmatched}, no_winner={no_winner})")
    return X, y, meta


def _lookup_player_stats(store, player_id, opponent_id, date_str, surface):
    """Look up player stats from stored rolling state."""
    elo = store["elo"].get(player_id, 1500.0)
    surface_elo_dict = store["surface_elo"].get(player_id, {})

    # Serve stats from history
    serve_history = store["serve_history"].get(player_id, [])
    serve_stats = _compute_serve_from_history(serve_history)

    # Form from history
    form_history = store["form_history"].get(player_id, [])
    form_stats = _compute_form_from_history(form_history, surface)

    # H2H
    h2h_key = tuple(sorted([player_id, opponent_id]))
    h2h_data = store["h2h"].get(h2h_key, {"wins": {}, "surface_wins": {}})
    h2h = {
        "wins": h2h_data.get("wins", {}).get(player_id, 0),
        "losses": h2h_data.get("wins", {}).get(opponent_id, 0),
        "surface_wins": dict(h2h_data.get("surface_wins", {}).get(player_id, {})),
        "surface_losses": dict(h2h_data.get("surface_wins", {}).get(opponent_id, {})),
    }

    # Physical
    player_info = store["players"].get(player_id, {})
    height = player_info.get("height", 0)
    if height == 0:
        height = 180.0
    hand = player_info.get("hand", "R")

    # Age
    dob = player_info.get("dob", "")
    age = 25.0
    if dob and len(dob) >= 8:
        try:
            from datetime import datetime
            dob_dt = datetime.strptime(dob[:8], "%Y%m%d")
            match_dt = datetime.strptime(date_str[:8], "%Y%m%d")
            age = (match_dt - dob_dt).days / 365.25
        except (ValueError, TypeError):
            pass

    # Rankings
    rankings_list = store["rankings"].get(player_id, [])
    from build_feature_store import get_ranking_at_date, get_rank_momentum, get_peak_rank
    rank, points = get_ranking_at_date(rankings_list, date_str)
    momentum = get_rank_momentum(rankings_list, date_str)
    peak = get_peak_rank(rankings_list, date_str)

    # Fatigue
    schedule = store["schedule"].get(player_id, [])
    fatigue = _compute_fatigue(schedule, date_str)

    return {
        "player_id": player_id,
        "elo": elo,
        "surface_elo": surface_elo_dict,
        "serve": serve_stats,
        "form": form_stats,
        "h2h": {opponent_id: h2h},
        "physical": {"height": height, "age": age, "hand": hand},
        "ranking": {"rank": rank, "points": points, "momentum": momentum, "peak": peak},
        "fatigue": fatigue,
    }


def _compute_serve_from_history(serve_history):
    """Compute serve stats from stored history list."""
    if not serve_history:
        return {
            "ace_rate": 0.05, "first_serve_pct": 0.60,
            "first_serve_won": 0.70, "second_serve_won": 0.50,
            "bp_save_rate": 0.60, "df_rate": 0.03,
            "serve_dominance": 0.0,
        }
    from collections import defaultdict
    total_weight = 0.0
    stats = defaultdict(float)
    for i, s in enumerate(reversed(serve_history[-20:])):
        w = 0.95 ** i
        total_weight += w
        for key in ["ace_rate", "first_serve_pct", "first_serve_won",
                    "second_serve_won", "bp_save_rate", "df_rate"]:
            stats[key] += s.get(key, 0.0) * w
    if total_weight > 0:
        for key in stats:
            stats[key] /= total_weight
    stats["serve_dominance"] = (
        stats["first_serve_won"] * 0.4 + stats["ace_rate"] * 0.3 - stats["df_rate"] * 0.3
    )
    return dict(stats)


def _compute_form_from_history(form_history, surface):
    """Compute form from stored history list."""
    if not form_history:
        return {"last_10": 0.5, "last_20": 0.5, "surface": {}, "weighted": 0.5,
                "upset_rate": 0.0, "momentum": 0.0}

    recent_10 = form_history[-10:]
    recent_20 = form_history[-20:]
    last_10 = sum(h[1] for h in recent_10) / len(recent_10)
    last_20 = sum(h[1] for h in recent_20) / len(recent_20)

    surface_form = {}
    for surf in ["Hard", "Clay", "Grass", "Carpet"]:
        s_matches = [h for h in form_history if h[2] == surf][-20:]
        surface_form[surf] = sum(h[1] for h in s_matches) / len(s_matches) if s_matches else 0.5

    weighted = 0.0
    tw = 0.0
    for i, h in enumerate(reversed(recent_20)):
        w = 0.95 ** i
        weighted += h[1] * w
        tw += w
    weighted = weighted / tw if tw > 0 else 0.5

    upsets = [h for h in recent_20 if h[3]]
    upset_rate = len(upsets) / max(len(recent_20), 1)

    momentum = 0.0
    if len(form_history) >= 10:
        last5 = sum(h[1] for h in form_history[-5:]) / 5
        prev5 = sum(h[1] for h in form_history[-10:-5]) / 5
        momentum = last5 - prev5

    return {"last_10": last_10, "last_20": last_20, "surface": surface_form,
            "weighted": weighted, "upset_rate": upset_rate, "momentum": momentum}


def _compute_fatigue(schedule, date_str):
    """Compute fatigue from schedule list."""
    from datetime import datetime, timedelta
    if not schedule or not date_str:
        return {"days_since_last": 14, "matches_14d": 3}
    try:
        dt = datetime.strptime(date_str[:8], "%Y%m%d")
    except ValueError:
        return {"days_since_last": 14, "matches_14d": 3}

    last_dates = []
    for d in schedule:
        if d < date_str[:8]:
            try:
                last_dates.append(datetime.strptime(d[:8], "%Y%m%d"))
            except ValueError:
                pass

    if last_dates:
        days_since = (dt - last_dates[-1]).days
    else:
        days_since = 30

    cutoff = dt - timedelta(days=14)
    matches_14d = sum(1 for d in last_dates if d >= cutoff)

    return {"days_since_last": days_since, "matches_14d": matches_14d}


def build_training_data():
    """Build and save training/test datasets."""
    print("Loading feature store...")
    with open(STORE_FILE, "rb") as f:
        store = pickle.load(f)
    print(f"  Loaded: {len(store['records'])} records")

    # Build historical samples
    print("\nBuilding historical samples...")
    hist_X, hist_y, hist_meta = build_historical_samples(store)

    # Build API samples
    print("\nBuilding API samples...")
    api_X, api_y, api_meta = build_api_samples(store)

    # Combine
    all_X = hist_X + api_X
    all_y = hist_y + api_y
    all_meta = hist_meta + api_meta

    print(f"\nTotal samples: {len(all_X)}")

    # Temporal split
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)

    dates = [m["date"] for m in all_meta]
    train_mask = np.array([d < "20260301" for d in dates])
    test_mask = np.array([d >= "20260301" for d in dates])

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Train: {len(X_train)} (< 2026-03-01)")
    print(f"Test:  {len(X_test)} (>= 2026-03-01)")
    print(f"Home win rate (train): {y_train.mean():.3f}")
    print(f"Home win rate (test):  {y_test.mean():.3f}")

    # Check for NaN
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"WARNING: {nan_count} NaN values in features, replacing with 0")
        X_train = np.nan_to_num(X_train, 0.0)
        X_test = np.nan_to_num(X_test, 0.0)

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.save(os.path.join(MODEL_DIR, "train_X.npy"), X_train)
    np.save(os.path.join(MODEL_DIR, "train_y.npy"), y_train)
    np.save(os.path.join(MODEL_DIR, "test_X.npy"), X_test)
    np.save(os.path.join(MODEL_DIR, "test_y.npy"), y_test)

    # Save meta
    train_meta = [m for m, mask in zip(all_meta, train_mask) if mask]
    test_meta = [m for m, mask in zip(all_meta, test_mask) if mask]
    with open(os.path.join(MODEL_DIR, "train_meta.json"), "w") as f:
        json.dump(train_meta, f)
    with open(os.path.join(MODEL_DIR, "test_meta.json"), "w") as f:
        json.dump(test_meta, f)

    # Save feature names
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(FEATURE_NAMES, f)

    print(f"\nSaved to model/train_X.npy, train_y.npy, test_X.npy, test_y.npy")
    print(f"Feature names: model/feature_names.json ({NUM_FEATURES} features)")

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    build_training_data()
