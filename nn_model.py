"""
Neural Network Tennis Match Predictor

Trains a neural network on historical match data with features extracted
from odds markets, then predicts upcoming matches.
"""

import json
import os
import glob
import pickle
import numpy as np
from datetime import datetime

# Feature extraction
def american_to_prob(odds_str):
    try:
        odds = int(odds_str)
    except (ValueError, TypeError):
        return None
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    elif odds > 0:
        return 100 / (odds + 100)
    return 0.5


def extract_features(event, use_opening=False):
    """
    Extract feature vector from a single event.

    Args:
        use_opening: If True, use openFairOdds/openBookOdds (pre-match).
    """
    odds = event.get("odds", {})
    if not odds:
        return None

    hml = odds.get("points-home-game-ml-home", {})
    aml = odds.get("points-away-game-ml-away", {})

    if not hml or not aml:
        return None

    # Core moneyline features - use opening or closing
    if use_opening:
        home_fair = american_to_prob(hml.get("openFairOdds"))
        away_fair = american_to_prob(aml.get("openFairOdds"))
        home_book = american_to_prob(hml.get("openBookOdds"))
        away_book = american_to_prob(aml.get("openBookOdds"))
    else:
        home_fair = american_to_prob(hml.get("fairOdds"))
        away_fair = american_to_prob(aml.get("fairOdds"))
        home_book = american_to_prob(hml.get("bookOdds"))
        away_book = american_to_prob(aml.get("bookOdds"))

    if home_fair is None or away_fair is None:
        return None

    # Normalize fair probs
    total_fair = home_fair + away_fair
    if total_fair > 0:
        home_fair_norm = home_fair / total_fair
        away_fair_norm = away_fair / total_fair
    else:
        return None

    # Vig
    home_book_safe = home_book if home_book else home_fair_norm
    away_book_safe = away_book if away_book else away_fair_norm
    vig_home = home_book_safe - home_fair_norm

    # Opening odds
    open_home_fair = american_to_prob(hml.get("openFairOdds"))
    if open_home_fair is not None:
        open_total = open_home_fair + (american_to_prob(aml.get("openFairOdds")) or (1 - open_home_fair))
        open_home_norm = open_home_fair / open_total if open_total > 0 else home_fair_norm
        movement = home_fair_norm - open_home_norm
    else:
        open_home_norm = home_fair_norm
        movement = 0.0

    # Bookmaker data
    bk_data = hml.get("byBookmaker", {})
    num_bookmakers = len(bk_data)
    bk_probs = []
    for bk, bd in bk_data.items():
        p = american_to_prob(bd.get("odds"))
        if p:
            bk_probs.append(p)

    if bk_probs:
        bk_spread = max(bk_probs) - min(bk_probs)
    else:
        bk_spread = 0.0

    # League
    league = 1.0 if event.get("leagueID") == "ATP" else 0.0

    # Favorite margin
    fav_margin = abs(home_fair_norm - away_fair_norm)

    # Helper: pick opening or closing odds for secondary markets
    def get_odds_val(market_key):
        mkt = odds.get(market_key, {})
        if use_opening:
            # Strict: only use opening odds, no fallback to closing
            return american_to_prob(mkt.get("openFairOdds"))
        return american_to_prob(mkt.get("fairOdds"))

    # Spread (games)
    games_sp_prob = get_odds_val("games-home-game-sp-home")
    if games_sp_prob is None:
        games_sp_prob = 0.5

    # Total games over/under
    total_ou_prob = get_odds_val("games-all-game-ou-over")
    if total_ou_prob is None:
        total_ou_prob = 0.5

    # 1st set moneyline
    set1_home_prob = get_odds_val("points-home-1s-ml-home")
    set1_away_prob = get_odds_val("points-away-1s-ml-away")
    if set1_home_prob is None:
        set1_home_prob = home_fair_norm
    if set1_away_prob is None:
        set1_away_prob = away_fair_norm
    s1_total = set1_home_prob + set1_away_prob
    if s1_total > 0:
        set1_home_prob /= s1_total
        set1_away_prob /= s1_total

    # Sets over/under
    sets_ou_prob = get_odds_val("points-all-game-ou-over")
    if sets_ou_prob is None:
        sets_ou_prob = 0.5

    # Player games
    home_games_prob = get_odds_val("games-home-game-ou-over")
    away_games_prob = get_odds_val("games-away-game-ou-over")
    if home_games_prob is None:
        home_games_prob = 0.5
    if away_games_prob is None:
        away_games_prob = 0.5
    player_games_diff = home_games_prob - away_games_prob

    features = np.array([
        home_fair_norm,         # 0
        away_fair_norm,         # 1
        home_book_safe,         # 2
        away_book_safe,         # 3
        vig_home,               # 4
        games_sp_prob,          # 5
        total_ou_prob,          # 6
        set1_home_prob,         # 7
        set1_away_prob,         # 8
        min(num_bookmakers / 30, 1.0),  # 9 normalized
        bk_spread,              # 10
        league,                 # 11
        fav_margin,             # 12
        open_home_norm,         # 13
        movement,               # 14
        games_sp_prob,          # 15
        total_ou_prob,          # 16
        player_games_diff,      # 17
    ], dtype=np.float32)

    return features


def determine_winner(event):
    """Return 1 if home won, 0 if away won, None if unknown."""
    teams = event.get("teams", {})
    results = event.get("results", {})

    home_score = teams.get("home", {}).get("score")
    away_score = teams.get("away", {}).get("score")

    if home_score is not None and away_score is not None:
        if home_score > away_score:
            return 1
        elif away_score > home_score:
            return 0

    game_results = results.get("game", {}) or results.get("reg", {})
    if game_results:
        home_pts = game_results.get("home", {}).get("points", 0)
        away_pts = game_results.get("away", {}).get("points", 0)
        if home_pts > away_pts:
            return 1
        elif away_pts > home_pts:
            return 0

    return None


def load_dataset():
    """Load all cached events and extract features + labels."""
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    files = sorted(glob.glob(os.path.join(cache_dir, "events_*.json")))

    X = []
    y = []
    meta = []

    # Two datasets: one with opening odds (honest), one with all data
    for filepath in files:
        with open(filepath) as f:
            events = json.load(f)

        for event in events:
            status = event.get("status", {})
            if not status.get("completed") or status.get("cancelled"):
                continue

            winner = determine_winner(event)
            if winner is None:
                continue

            odds_obj = event.get("odds", {})
            hml = odds_obj.get("points-home-game-ml-home", {})
            has_opening = bool(hml.get("openFairOdds"))

            # Use opening odds where available, closing otherwise
            features = extract_features(event, use_opening=has_opening)
            if features is None:
                continue

            X.append(features)
            y.append(winner)
            meta.append({
                "home": event["teams"]["home"]["names"]["long"],
                "away": event["teams"]["away"]["names"]["long"],
                "league": event.get("leagueID"),
                "date": status.get("startsAt", "")[:10],
                "has_opening": has_opening,
            })

    return np.array(X), np.array(y), meta


def train_model():
    """Train the neural network and return results."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print("Loading dataset...")
    X, y, meta = load_dataset()
    print(f"Dataset: {len(X)} matches, {sum(y)} home wins ({sum(y)/len(y)*100:.1f}%)")

    # Split: train on data before Feb 2026 (mostly closing odds, big dataset)
    # Test on Feb-Mar 2026 data that has opening odds (honest evaluation)
    opening_mask = np.array([m["has_opening"] for m in meta])
    date_mask = np.array([m["date"] >= "2026-02-01" for m in meta])

    # Train: everything that ISN'T in the honest test set
    honest_test_mask = opening_mask & date_mask
    train_mask = ~honest_test_mask

    # Also do a time-based honest split: train on opening odds before Mar,
    # test on March opening odds
    march_mask = np.array([m["date"] >= "2026-03-01" for m in meta])
    feb_opening_mask = opening_mask & date_mask & ~march_mask
    mar_opening_mask = opening_mask & march_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[honest_test_mask], y[honest_test_mask]
    X_honest, y_honest = X[mar_opening_mask], y[mar_opening_mask]

    print(f"Train: {len(X_train)} (mostly closing odds)")
    print(f"Test (all opening odds): {len(X_test)}")
    print(f"Test (March only, strictest): {len(X_honest)}")

    # Normalize features
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1  # avoid division by zero
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    X_honest_norm = (X_honest - mean) / std

    # PyTorch model
    class TennisNet(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    model = TennisNet(X_train_norm.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCELoss()

    # Convert to tensors
    X_t = torch.FloatTensor(X_train_norm)
    y_t = torch.FloatTensor(y_train).unsqueeze(1)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Train
    print("\nTraining neural network...")
    best_loss = float("inf")
    patience = 0

    for epoch in range(200):
        model.train()
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(torch.FloatTensor(X_test_norm))
                test_acc = ((test_pred.numpy().flatten() > 0.5) == y_test).mean()
                if len(X_honest) > 0:
                    honest_pred = model(torch.FloatTensor(X_honest_norm))
                    honest_acc = ((honest_pred.numpy().flatten() > 0.5) == y_honest).mean()
                else:
                    honest_acc = 0
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Test acc: {test_acc*100:.1f}% | Honest acc: {honest_acc*100:.1f}%")

        # Early stopping
        if avg_loss < best_loss - 0.001:
            best_loss = avg_loss
            patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Final evaluation
    with torch.no_grad():
        # All 2026 test
        test_probs = model(torch.FloatTensor(X_test_norm)).numpy().flatten()
        test_preds = test_probs > 0.5
        test_acc = (test_preds == y_test).mean()

        # Honest test (opening odds only)
        if len(X_honest) > 0:
            honest_probs = model(torch.FloatTensor(X_honest_norm)).numpy().flatten()
            honest_preds = honest_probs > 0.5
            honest_acc = (honest_preds == y_honest).mean()
        else:
            honest_acc = 0
            honest_probs = np.array([])

        # Train accuracy
        train_probs = model(torch.FloatTensor(X_train_norm)).numpy().flatten()
        train_acc = ((train_probs > 0.5) == y_train).mean()

    # Baseline comparison (just using fair odds prob > 0.5)
    baseline_test = (X_test[:, 0] > 0.5) == y_test
    baseline_acc = baseline_test.mean()
    if len(X_honest) > 0:
        baseline_honest = (X_honest[:, 0] > 0.5) == y_honest
        baseline_honest_acc = baseline_honest.mean()
    else:
        baseline_honest_acc = 0

    # Calibration
    calibration = {}
    for bucket_name, lo, hi in [("50-60", 0.5, 0.6), ("60-70", 0.6, 0.7),
                                  ("70-80", 0.7, 0.8), ("80-90", 0.8, 0.9),
                                  ("90-100", 0.9, 1.01)]:
        # Use winner prob (max of home/away prob)
        winner_probs = np.maximum(test_probs, 1 - test_probs)
        mask = (winner_probs >= lo) & (winner_probs < hi)
        if mask.sum() > 0:
            # Did the predicted winner actually win?
            predicted_home = test_probs > 0.5
            correct = predicted_home == y_test
            calibration[bucket_name] = {
                "total": int(mask.sum()),
                "correct": int(correct[mask].sum()),
                "actual_pct": round(correct[mask].mean() * 100, 1),
                "avg_predicted_pct": round(winner_probs[mask].mean() * 100, 1),
            }
        else:
            calibration[bucket_name] = {"total": 0, "correct": 0, "actual_pct": 0, "avg_predicted_pct": 0}

    results = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "honest_test_size": len(X_honest),
        "train_acc": round(train_acc * 100, 1),
        "test_acc": round(test_acc * 100, 1),
        "honest_acc": round(honest_acc * 100, 1),
        "baseline_acc": round(baseline_acc * 100, 1),
        "baseline_honest_acc": round(baseline_honest_acc * 100, 1),
        "improvement": round((honest_acc - baseline_honest_acc) * 100, 1) if len(X_honest) > 0 else 0,
        "calibration": calibration,
        "num_features": X_train.shape[1],
    }

    # Save model + normalization params
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "tennis_nn.pt"))
    with open(os.path.join(model_dir, "norm_params.pkl"), "wb") as f:
        pickle.dump({"mean": mean, "std": std, "input_size": X_train.shape[1]}, f)

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Train accuracy:           {results['train_acc']}% ({len(X_train)} matches)")
    print(f"Test accuracy (all 2026): {results['test_acc']}% ({len(X_test)} matches)")
    print(f"Test accuracy (opening):  {results['honest_acc']}% ({len(X_honest)} matches)")
    print(f"Baseline (fair odds):     {results['baseline_honest_acc']}% (opening odds)")
    print(f"NN improvement:           {results['improvement']:+.1f}%")
    print(f"\nCalibration:")
    for bucket, data in calibration.items():
        if data["total"] > 0:
            print(f"  {bucket}%: pred {data['avg_predicted_pct']}% -> actual {data['actual_pct']}% ({data['total']})")
    print(f"\nModel saved to model/tennis_nn.pt")

    return results, model, mean, std


def predict_match(event, model=None, mean=None, std=None):
    """Predict a single match using the trained NN."""
    import torch
    import torch.nn as nn

    # Load model if not provided
    if model is None:
        model_dir = os.path.join(os.path.dirname(__file__), "model")
        with open(os.path.join(model_dir, "norm_params.pkl"), "rb") as f:
            params = pickle.load(f)
        mean = params["mean"]
        std = params["std"]
        input_size = params["input_size"]

        class TennisNet(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_size, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                    nn.Sigmoid(),
                )
            def forward(self, x):
                return self.net(x)

        model = TennisNet(input_size)
        model.load_state_dict(torch.load(os.path.join(model_dir, "tennis_nn.pt")))
        model.eval()

    features = extract_features(event)
    if features is None:
        return None

    features_norm = (features - mean) / std
    with torch.no_grad():
        prob = model(torch.FloatTensor(features_norm).unsqueeze(0)).item()

    return {
        "home_win_prob": round(prob * 100, 1),
        "away_win_prob": round((1 - prob) * 100, 1),
        "predicted_winner": "home" if prob > 0.5 else "away",
        "confidence": round(abs(prob - 0.5) * 200, 1),
    }


if __name__ == "__main__":
    results, model, mean, std = train_model()
