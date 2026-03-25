"""
Deep Learning Tennis Match Predictor

Architecture:
1. Player Embeddings — learned vector per player (captures style/strength)
2. LSTM Sequence — processes each player's recent match history (form/momentum)
3. Match Features — odds/market signals from current match
4. Deep Fusion — combines all signals through deeper layers

Inspired by Chapter 6 of "Neural Networks and Deep Learning" (Michael Nielsen)
"""

import json
import os
import glob
import pickle
import numpy as np
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from nn_model import american_to_prob, extract_features, determine_winner

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

SEQUENCE_LENGTH = 10  # last N matches per player
MATCH_FEATURES = 18   # from extract_features()


def build_player_history():
    """
    Build per-player match history from cached data.
    Returns dict: player_id -> list of {date, won, odds_prob, opponent, league, ...}
    """
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "events_*.json")))
    history = defaultdict(list)
    all_events = []

    for filepath in files:
        with open(filepath) as f:
            events = json.load(f)
        for e in events:
            s = e.get("status", {})
            if not s.get("completed") or s.get("cancelled"):
                continue
            all_events.append(e)

    # Sort by date
    all_events.sort(key=lambda e: e["status"].get("startsAt", ""))

    for event in all_events:
        home_id = event["teams"]["home"].get("teamID", "")
        away_id = event["teams"]["away"].get("teamID", "")
        if not home_id or not away_id:
            continue

        winner = determine_winner(event)
        if winner is None:
            continue

        date = event["status"].get("startsAt", "")[:10]
        league = event.get("leagueID", "")

        # Get odds
        odds = event.get("odds", {})
        hml = odds.get("points-home-game-ml-home", {})
        aml = odds.get("points-away-game-ml-away", {})

        # Use opening if available, else closing
        h_fair = hml.get("openFairOdds") or hml.get("fairOdds")
        a_fair = aml.get("openFairOdds") or aml.get("fairOdds")
        h_prob = american_to_prob(h_fair) or 0.5
        a_prob = american_to_prob(a_fair) or 0.5

        # Normalize
        total = h_prob + a_prob
        h_prob /= total
        a_prob /= total

        # Home player record
        history[home_id].append({
            "date": date,
            "won": 1.0 if winner == 1 else 0.0,
            "my_prob": h_prob,
            "opp_prob": a_prob,
            "opponent": away_id,
            "league": league,
            "was_favorite": 1.0 if h_prob > 0.5 else 0.0,
            "margin": h_prob - a_prob,
            "event": event,
        })

        # Away player record
        history[away_id].append({
            "date": date,
            "won": 1.0 if winner == 0 else 0.0,
            "my_prob": a_prob,
            "opp_prob": h_prob,
            "opponent": home_id,
            "league": league,
            "was_favorite": 1.0 if a_prob > 0.5 else 0.0,
            "margin": a_prob - h_prob,
            "event": event,
        })

    return history, all_events


def build_player_index(history, min_matches=5):
    """Create player ID -> index mapping for embeddings."""
    player_ids = sorted([pid for pid, h in history.items() if len(h) >= min_matches])
    player_to_idx = {pid: i + 1 for i, pid in enumerate(player_ids)}  # 0 = unknown
    return player_to_idx


def get_player_sequence(history, player_id, before_date, seq_len=SEQUENCE_LENGTH):
    """
    Get a player's last N matches before a given date.
    Returns tensor of shape (seq_len, features_per_step).

    Features per step:
    0: won (0/1)
    1: my implied probability
    2: was favorite (0/1)
    3: margin (my_prob - opp_prob)
    4: win streak momentum (rolling avg of last 5 wins)
    """
    matches = history.get(player_id, [])
    recent = [m for m in matches if m["date"] < before_date]
    recent = recent[-seq_len:]  # last N

    seq = np.zeros((seq_len, 5), dtype=np.float32)

    for i, m in enumerate(recent):
        offset = seq_len - len(recent) + i
        seq[offset, 0] = m["won"]
        seq[offset, 1] = m["my_prob"]
        seq[offset, 2] = m["was_favorite"]
        seq[offset, 3] = m["margin"]

        # Rolling win rate (last 5 before this match)
        prev = recent[max(0, i - 5):i]
        if prev:
            seq[offset, 4] = sum(p["won"] for p in prev) / len(prev)
        else:
            seq[offset, 4] = 0.5

    return seq


class TennisDeepDataset(Dataset):
    """Dataset that yields (home_seq, away_seq, home_id, away_id, match_features, label)."""

    def __init__(self, events, history, player_to_idx, use_opening=True):
        self.samples = []

        for event in events:
            home_id = event["teams"]["home"].get("teamID", "")
            away_id = event["teams"]["away"].get("teamID", "")
            if not home_id or not away_id:
                continue

            winner = determine_winner(event)
            if winner is None:
                continue

            hml = event.get("odds", {}).get("points-home-game-ml-home", {})
            has_opening = bool(hml.get("openFairOdds"))
            features = extract_features(event, use_opening=(use_opening and has_opening))
            if features is None:
                continue

            date = event["status"].get("startsAt", "")[:10]
            home_seq = get_player_sequence(history, home_id, date)
            away_seq = get_player_sequence(history, away_id, date)
            home_idx = player_to_idx.get(home_id, 0)
            away_idx = player_to_idx.get(away_id, 0)

            self.samples.append({
                "home_seq": torch.FloatTensor(home_seq),
                "away_seq": torch.FloatTensor(away_seq),
                "home_idx": home_idx,
                "away_idx": away_idx,
                "features": torch.FloatTensor(features),
                "label": float(winner),
                "has_opening": has_opening,
                "date": date,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s["home_seq"], s["away_seq"],
            s["home_idx"], s["away_idx"],
            s["features"], s["label"],
        )


class TennisDeepNet(nn.Module):
    """
    Deep learning model combining:
    1. Player embeddings (learned strength vectors)
    2. LSTM for recent match sequences (form/momentum)
    3. Match-level features (odds, spreads, O/U)
    4. Deep fusion network
    """

    def __init__(self, num_players, embed_dim=32, lstm_hidden=32,
                 match_features=MATCH_FEATURES):
        super().__init__()

        # Player embeddings
        self.player_embed = nn.Embedding(num_players + 1, embed_dim, padding_idx=0)

        # LSTM for match sequences (shared weights for both players)
        self.sequence_lstm = nn.LSTM(
            input_size=5,        # features per time step
            hidden_size=lstm_hidden,
            num_layers=2,        # deep LSTM
            batch_first=True,
            dropout=0.2,
            bidirectional=False,
        )

        # Match features encoder
        self.match_encoder = nn.Sequential(
            nn.Linear(match_features, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 32),
            nn.ReLU(),
        )

        # Fusion: combine all signals
        # 2 * embed_dim (home + away) + 2 * lstm_hidden (home + away seq) + 32 (match features)
        fusion_input = 2 * embed_dim + 2 * lstm_hidden + 32

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, home_seq, away_seq, home_idx, away_idx, match_features):
        # Player embeddings
        home_emb = self.player_embed(home_idx)   # (batch, embed_dim)
        away_emb = self.player_embed(away_idx)

        # LSTM on sequences
        home_lstm_out, _ = self.sequence_lstm(home_seq)  # (batch, seq_len, hidden)
        away_lstm_out, _ = self.sequence_lstm(away_seq)

        # Take last hidden state
        home_form = home_lstm_out[:, -1, :]  # (batch, hidden)
        away_form = away_lstm_out[:, -1, :]

        # Match features
        match_enc = self.match_encoder(match_features)  # (batch, 32)

        # Fuse everything
        combined = torch.cat([
            home_emb, away_emb,
            home_form, away_form,
            match_enc
        ], dim=1)

        return self.fusion(combined)


def train_deep_model():
    """Train the deep learning model."""
    print("Building player histories...")
    history, all_events = build_player_history()
    player_to_idx = build_player_index(history, min_matches=5)
    print(f"Players with embeddings: {len(player_to_idx)}")
    print(f"Total events: {len(all_events)}")

    # Split: train on pre-2026, test on 2026 with opening odds
    train_events = [e for e in all_events if e["status"].get("startsAt", "")[:10] < "2026-02-01"]
    test_events = [e for e in all_events if e["status"].get("startsAt", "")[:10] >= "2026-02-01"]

    print(f"Train events: {len(train_events)}")
    print(f"Test events: {len(test_events)}")

    train_ds = TennisDeepDataset(train_events, history, player_to_idx, use_opening=True)
    test_ds = TennisDeepDataset(test_events, history, player_to_idx, use_opening=True)

    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")

    # Normalize match features using training stats
    all_feats = torch.stack([s["features"] for s in train_ds.samples])
    feat_mean = all_feats.mean(0)
    feat_std = all_feats.std(0)
    feat_std[feat_std == 0] = 1

    for s in train_ds.samples:
        s["features"] = (s["features"] - feat_mean) / feat_std
    for s in test_ds.samples:
        s["features"] = (s["features"] - feat_mean) / feat_std

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Model
    model = TennisDeepNet(num_players=len(player_to_idx), embed_dim=32, lstm_hidden=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"  Embeddings: {sum(p.numel() for p in model.player_embed.parameters()):,}")
    print(f"  LSTM: {sum(p.numel() for p in model.sequence_lstm.parameters()):,}")
    print(f"  Match encoder: {sum(p.numel() for p in model.match_encoder.parameters()):,}")
    print(f"  Fusion: {sum(p.numel() for p in model.fusion.parameters()):,}")

    print("\nTraining deep model...")
    best_acc = 0
    best_state = None
    patience = 0

    for epoch in range(150):
        model.train()
        total_loss = 0
        for batch in train_loader:
            home_seq, away_seq, home_idx, away_idx, features, labels = batch
            optimizer.zero_grad()
            pred = model(home_seq, away_seq, home_idx.long(), away_idx.long(), features)
            loss = criterion(pred.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        if epoch % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            opening_correct = 0
            opening_total = 0

            with torch.no_grad():
                for batch in test_loader:
                    home_seq, away_seq, home_idx, away_idx, features, labels = batch
                    pred = model(home_seq, away_seq, home_idx.long(), away_idx.long(), features)
                    predicted = (pred.squeeze() > 0.5).float()
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            acc = correct / total if total > 0 else 0

            # Opening odds only accuracy
            for s in test_ds.samples:
                if s["has_opening"]:
                    opening_total += 1
                    # Would need to re-run predictions here, use overall for now

            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Test acc: {acc*100:.1f}% ({correct}/{total})")

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 8:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    # Load best
    model.load_state_dict(best_state)
    model.eval()

    # Final detailed evaluation
    print(f"\n{'='*50}")
    print("FINAL EVALUATION")
    print(f"{'='*50}")

    correct = 0
    total = 0
    opening_correct = 0
    opening_total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for s in test_ds.samples:
            pred = model(
                s["home_seq"].unsqueeze(0), s["away_seq"].unsqueeze(0),
                torch.LongTensor([s["home_idx"]]), torch.LongTensor([s["away_idx"]]),
                s["features"].unsqueeze(0)
            ).item()

            label = s["label"]
            is_correct = (pred > 0.5) == (label > 0.5)
            total += 1
            if is_correct:
                correct += 1

            if s["has_opening"]:
                opening_total += 1
                if is_correct:
                    opening_correct += 1

            all_probs.append(pred)
            all_labels.append(label)

    # Baseline: simple fair odds
    baseline_correct = sum(
        1 for s in test_ds.samples
        if (s["features"][0].item() > 0) == (s["label"] > 0.5)  # feature 0 normalized
    )

    acc = correct / total * 100 if total else 0
    opening_acc = opening_correct / opening_total * 100 if opening_total else 0
    baseline_acc = baseline_correct / total * 100 if total else 0

    print(f"Test accuracy (all):     {acc:.1f}% ({correct}/{total})")
    print(f"Test accuracy (opening): {opening_acc:.1f}% ({opening_correct}/{opening_total})")
    print(f"Baseline (odds model):   {baseline_acc:.1f}%")
    print(f"Deep learning improvement: {acc - baseline_acc:+.1f}%")

    # Calibration
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    winner_probs = np.maximum(probs, 1 - probs)
    predicted_correct = ((probs > 0.5) == (labels > 0.5))

    print(f"\nCalibration:")
    for name, lo, hi in [("50-60", 0.5, 0.6), ("60-70", 0.6, 0.7),
                          ("70-80", 0.7, 0.8), ("80-90", 0.8, 0.9), ("90-100", 0.9, 1.01)]:
        mask = (winner_probs >= lo) & (winner_probs < hi)
        if mask.sum() > 0:
            actual = predicted_correct[mask].mean() * 100
            pred_avg = winner_probs[mask].mean() * 100
            print(f"  {name}%: pred {pred_avg:.1f}% -> actual {actual:.1f}% ({mask.sum()} matches)")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "tennis_deep.pt"))
    with open(os.path.join(MODEL_DIR, "deep_params.pkl"), "wb") as f:
        pickle.dump({
            "player_to_idx": player_to_idx,
            "feat_mean": feat_mean.numpy(),
            "feat_std": feat_std.numpy(),
            "num_players": len(player_to_idx),
        }, f)

    print(f"\nDeep model saved to model/tennis_deep.pt")
    print(f"Player embeddings: {len(player_to_idx)} players")

    return model


if __name__ == "__main__":
    train_deep_model()
