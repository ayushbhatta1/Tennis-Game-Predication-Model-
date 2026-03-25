"""
Historical Tennis Model — trained on 143K+ matches (2000-2024)

Uses Jeff Sackmann's open dataset with player rankings, stats,
surface, round, seeds, serve stats, and computed features like
ELO ratings and recent form.
"""

import csv
import os
import glob
import pickle
import numpy as np
from collections import defaultdict

HIST_DIR = os.path.join(os.path.dirname(__file__), "historical")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


def compute_elo(matches, k=32):
    """Compute ELO ratings for all players from match history."""
    elo = defaultdict(lambda: 1500.0)

    for m in matches:
        w_id = m["winner_id"]
        l_id = m["loser_id"]

        w_elo = elo[w_id]
        l_elo = elo[l_id]

        # Expected scores
        exp_w = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
        exp_l = 1 - exp_w

        # Update
        elo[w_id] = w_elo + k * (1 - exp_w)
        elo[l_id] = l_elo + k * (0 - exp_l)

        m["w_elo"] = w_elo  # ELO at time of match (before update)
        m["l_elo"] = l_elo

    return elo


def compute_form(matches, window=10):
    """Compute recent win rate for each player."""
    history = defaultdict(list)  # player_id -> list of (date, won)

    for m in matches:
        w_id = m["winner_id"]
        l_id = m["loser_id"]
        date = m["tourney_date"]

        # Winner's form at this point
        recent_w = history[w_id][-window:]
        m["w_form"] = sum(r[1] for r in recent_w) / len(recent_w) if recent_w else 0.5

        recent_l = history[l_id][-window:]
        m["l_form"] = sum(r[1] for r in recent_l) / len(recent_l) if recent_l else 0.5

        # Update history
        history[w_id].append((date, 1))
        history[l_id].append((date, 0))


def compute_h2h(matches):
    """Compute head-to-head records."""
    h2h = defaultdict(lambda: [0, 0])  # (p1, p2) -> [p1_wins, p2_wins]

    for m in matches:
        w_id = m["winner_id"]
        l_id = m["loser_id"]

        key = tuple(sorted([w_id, l_id]))
        record = h2h[key]

        if key[0] == w_id:
            m["h2h_w"] = record[0]
            m["h2h_l"] = record[1]
            record[0] += 1
        else:
            m["h2h_w"] = record[1]
            m["h2h_l"] = record[0]
            record[1] += 1


def compute_surface_form(matches, window=20):
    """Compute surface-specific win rate."""
    history = defaultdict(lambda: defaultdict(list))

    for m in matches:
        surface = m.get("surface", "Hard")
        w_id = m["winner_id"]
        l_id = m["loser_id"]

        recent_w = history[w_id][surface][-window:]
        m["w_surface_form"] = sum(r for r in recent_w) / len(recent_w) if recent_w else 0.5

        recent_l = history[l_id][surface][-window:]
        m["l_surface_form"] = sum(r for r in recent_l) / len(recent_l) if recent_l else 0.5

        history[w_id][surface].append(1)
        history[l_id][surface].append(0)


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def load_all_matches():
    """Load all ATP + WTA matches, sorted by date."""
    matches = []

    for pattern in ["atp_matches_*.csv", "wta_matches_*.csv"]:
        files = sorted(glob.glob(os.path.join(HIST_DIR, pattern)))
        for filepath in files:
            league = "ATP" if "atp_" in filepath else "WTA"
            with open(filepath, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["league"] = league
                    row["winner_id"] = row.get("winner_id", "")
                    row["loser_id"] = row.get("loser_id", "")
                    row["tourney_date"] = row.get("tourney_date", "")
                    matches.append(row)

    # Sort by date
    matches.sort(key=lambda m: m["tourney_date"])
    return matches


def extract_historical_features(m):
    """Extract feature vector from a historical match."""
    # Rankings
    w_rank = safe_float(m.get("winner_rank"), 500)
    l_rank = safe_float(m.get("loser_rank"), 500)
    w_pts = safe_float(m.get("winner_rank_points"), 0)
    l_pts = safe_float(m.get("loser_rank_points"), 0)

    # Rank features (log-scaled)
    rank_diff = np.log1p(l_rank) - np.log1p(w_rank)  # positive = winner ranked higher
    pts_ratio = w_pts / (w_pts + l_pts) if (w_pts + l_pts) > 0 else 0.5

    # Seeds
    w_seed = safe_float(m.get("winner_seed"), 33)
    l_seed = safe_float(m.get("loser_seed"), 33)
    seed_diff = l_seed - w_seed

    # Physical
    w_ht = safe_float(m.get("winner_ht"), 180)
    l_ht = safe_float(m.get("loser_ht"), 180)
    w_age = safe_float(m.get("winner_age"), 25)
    l_age = safe_float(m.get("loser_age"), 25)

    # Surface encoding
    surface = m.get("surface", "Hard")
    surface_hard = 1.0 if surface == "Hard" else 0.0
    surface_clay = 1.0 if surface == "Clay" else 0.0
    surface_grass = 1.0 if surface == "Grass" else 0.0

    # Round encoding
    round_map = {"F": 7, "SF": 6, "QF": 5, "R16": 4, "R32": 3, "R64": 2, "R128": 1, "RR": 3}
    round_val = round_map.get(m.get("round", ""), 3) / 7.0

    # Tournament level
    level = m.get("tourney_level", "A")
    level_gs = 1.0 if level == "G" else 0.0
    level_masters = 1.0 if level == "M" else 0.0

    # Best of
    best_of = safe_float(m.get("best_of"), 3) / 5.0

    # League
    league_atp = 1.0 if m.get("league") == "ATP" else 0.0

    # ELO (computed earlier)
    w_elo = m.get("w_elo", 1500)
    l_elo = m.get("l_elo", 1500)
    elo_diff = (w_elo - l_elo) / 400.0  # normalized

    # Form (computed earlier)
    w_form = m.get("w_form", 0.5)
    l_form = m.get("l_form", 0.5)

    # Surface form
    w_surf = m.get("w_surface_form", 0.5)
    l_surf = m.get("l_surface_form", 0.5)

    # H2H
    h2h_w = m.get("h2h_w", 0)
    h2h_l = m.get("h2h_l", 0)
    h2h_total = h2h_w + h2h_l
    h2h_ratio = h2h_w / h2h_total if h2h_total > 0 else 0.5

    features = np.array([
        rank_diff,          # 0: log rank difference
        pts_ratio,          # 1: ranking points ratio
        seed_diff,          # 2: seed difference
        w_ht - l_ht,        # 3: height difference
        w_age - l_age,      # 4: age difference
        surface_hard,       # 5
        surface_clay,       # 6
        surface_grass,      # 7
        round_val,          # 8: round depth
        level_gs,           # 9: grand slam
        level_masters,      # 10: masters
        best_of,            # 11
        league_atp,         # 12
        elo_diff,           # 13: ELO difference
        w_form - l_form,    # 14: recent form difference
        w_surf - l_surf,    # 15: surface form difference
        h2h_ratio,          # 16: head-to-head ratio
        h2h_total / 20.0,   # 17: h2h familiarity (normalized)
        np.log1p(w_rank) / 7, # 18: winner rank (normalized)
        np.log1p(l_rank) / 7, # 19: loser rank (normalized)
    ], dtype=np.float32)

    return features


def build_dataset():
    """Build full training dataset with computed features."""
    print("Loading matches...")
    matches = load_all_matches()
    print(f"Total matches: {len(matches)}")

    print("Computing ELO ratings...")
    compute_elo(matches)

    print("Computing recent form...")
    compute_form(matches)

    print("Computing surface form...")
    compute_surface_form(matches)

    print("Computing head-to-head...")
    compute_h2h(matches)

    print("Extracting features...")
    X = []
    y = []
    meta = []

    for m in matches:
        if not m.get("winner_id") or not m.get("loser_id"):
            continue

        features = extract_historical_features(m)

        # Randomly assign winner to home/away to avoid position bias
        # Use hash of match for deterministic randomness
        match_hash = hash(m["tourney_id"] + str(m["match_num"]))
        if match_hash % 2 == 0:
            # Winner is "home"
            X.append(features)
            y.append(1)
        else:
            # Winner is "away" — flip all relative features
            flipped = features.copy()
            flipped[0] = -flipped[0]   # rank diff
            flipped[1] = 1 - flipped[1]  # pts ratio
            flipped[2] = -flipped[2]   # seed diff
            flipped[3] = -flipped[3]   # height diff
            flipped[4] = -flipped[4]   # age diff
            flipped[13] = -flipped[13]  # elo diff
            flipped[14] = -flipped[14]  # form diff
            flipped[15] = -flipped[15]  # surface form diff
            flipped[16] = 1 - flipped[16]  # h2h ratio
            flipped[18], flipped[19] = flipped[19], flipped[18]  # swap ranks
            X.append(flipped)
            y.append(0)

        meta.append({
            "date": m["tourney_date"],
            "league": m.get("league"),
            "winner": m.get("winner_name", ""),
            "loser": m.get("loser_name", ""),
            "surface": m.get("surface", ""),
        })

    return np.array(X), np.array(y), meta


def train():
    """Train XGBoost on 25 years of historical data."""
    import xgboost as xgb

    X, y, meta = build_dataset()
    print(f"\nDataset: {len(X)} matches, {X.shape[1]} features")
    print(f"Home wins: {sum(y)}/{len(y)} ({sum(y)/len(y)*100:.1f}%)")

    # Time-based splits
    dates = np.array([m["date"] for m in meta])

    # Train: 2000-2022, Validate: 2023, Test: 2024
    train_mask = dates < "20230101"
    val_mask = (dates >= "20230101") & (dates < "20240101")
    test_mask = dates >= "20240101"

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    print(f"Train: {len(X_tr)} (2000-2022)")
    print(f"Val: {len(X_val)} (2023)")
    print(f"Test: {len(X_te)} (2024)")

    # Grid search on validation set
    print("\nGrid search...")
    best_acc = 0
    best_params = None

    for max_depth in [4, 5, 6]:
        for lr in [0.05, 0.1]:
            for n_est in [200, 300, 500]:
                model = xgb.XGBClassifier(
                    max_depth=max_depth, learning_rate=lr, n_estimators=n_est,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42, eval_metric='logloss',
                )
                model.fit(X_tr, y_tr, verbose=False)
                val_acc = (model.predict(X_val) == y_val).mean()
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_params = {"max_depth": max_depth, "lr": lr, "n_est": n_est}

    print(f"Best val params: {best_params} ({best_acc*100:.1f}%)")

    # Train final model on train+val, test on 2024
    X_trainval = np.vstack([X_tr, X_val])
    y_trainval = np.concatenate([y_tr, y_val])

    model = xgb.XGBClassifier(
        max_depth=best_params["max_depth"],
        learning_rate=best_params["lr"],
        n_estimators=best_params["n_est"],
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, eval_metric='logloss',
    )
    model.fit(X_trainval, y_trainval, verbose=False)

    test_acc = (model.predict(X_te) == y_te).mean()
    test_probs = model.predict_proba(X_te)[:, 1]

    # Feature importance
    feat_names = [
        "rank_diff", "pts_ratio", "seed_diff", "height_diff", "age_diff",
        "hard", "clay", "grass", "round", "grand_slam", "masters",
        "best_of", "league", "elo_diff", "form_diff", "surface_form_diff",
        "h2h_ratio", "h2h_familiar", "rank_a", "rank_b"
    ]
    fi = model.feature_importances_

    # Calibration
    winner_probs = np.maximum(test_probs, 1 - test_probs)
    correct = (model.predict(X_te) == y_te)

    print(f"\n{'='*55}")
    print(f"  HISTORICAL MODEL — 25 YEARS (2000-2024)")
    print(f"{'='*55}")
    print(f"  Train: {len(X_trainval)} matches (2000-2023)")
    print(f"  Test:  {len(X_te)} matches (2024)")
    print(f"  Test accuracy: {test_acc*100:.1f}%")
    print(f"")
    print(f"  Top features:")
    for i in np.argsort(fi)[::-1][:8]:
        print(f"    {feat_names[i]:22s} {fi[i]:.3f}")
    print(f"\n  Calibration:")
    for name, lo, hi in [("50-60", 0.5, 0.6), ("60-70", 0.6, 0.7),
                          ("70-80", 0.7, 0.8), ("80-90", 0.8, 0.9), ("90-100", 0.9, 1.01)]:
        mask = (winner_probs >= lo) & (winner_probs < hi)
        if mask.sum() > 0:
            print(f"    {name}%: pred {winner_probs[mask].mean()*100:.1f}% -> actual {correct[mask].mean()*100:.1f}% ({mask.sum()})")

    # Save production model (trained on ALL data)
    print(f"\nTraining production model on all {len(X)} matches...")
    prod_model = xgb.XGBClassifier(
        max_depth=best_params["max_depth"],
        learning_rate=best_params["lr"],
        n_estimators=best_params["n_est"],
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, eval_metric='logloss',
    )
    prod_model.fit(X, y, verbose=False)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "historical_xgb.pkl"), "wb") as f:
        pickle.dump(prod_model, f)
    print("Saved to model/historical_xgb.pkl")

    return model, test_acc


if __name__ == "__main__":
    train()
