"""
Mega Deep LSTM Trainer — combines player embeddings, LSTM sequences,
attention, and the full 66-feature unified feature engine.

Architecture:
  1. Player embeddings (64-dim, up to 35000 players)
  2. LSTM (2-layer, hidden=64) over 15-step match history per player
  3. Attention mechanism over LSTM hidden states
  4. Match features encoder (66 -> 96 -> 64)
  5. Residual fusion network (320 -> 256 -> 128 -> 64 -> 1)

Data:
  - model/feature_store.pkl for player histories + records
  - model/train_X.npy, train_y.npy, test_X.npy, test_y.npy for match features
  - model/train_meta.json, test_meta.json for dates/player IDs
  - player_resolver for API team-ID -> Sackmann ID mapping

Output:
  - model/mega_deep.pt  (state dict)
  - model/mega_deep_params.pkl  (player_to_idx, feat_mean, feat_std, etc.)
"""

import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

SEQUENCE_LENGTH = 15
LSTM_INPUT_SIZE = 10  # per step features
NUM_MATCH_FEATURES = 66


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MegaDeepNet(nn.Module):
    """
    Deep LSTM model with:
      - Player embeddings
      - 2-layer LSTM over 15-step player history
      - Attention over LSTM hidden states
      - 66-feature match encoder
      - Residual fusion network
    """

    def __init__(self, num_players, embed_dim=64, lstm_hidden=64,
                 match_features=NUM_MATCH_FEATURES):
        super().__init__()

        # --- Player embeddings ---
        self.player_embed = nn.Embedding(num_players + 1, embed_dim, padding_idx=0)

        # --- Shared LSTM for player sequences ---
        self.sequence_lstm = nn.LSTM(
            input_size=LSTM_INPUT_SIZE,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
        )

        # --- Attention over LSTM hidden states ---
        self.attention = nn.Linear(lstm_hidden, 1)

        # --- Match features encoder ---
        self.match_encoder = nn.Sequential(
            nn.Linear(match_features, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(96, 64),
            nn.ReLU(),
        )

        # --- Fusion with residual ---
        # Input: home_emb(64) + away_emb(64) + home_attn(64) + away_attn(64) + match_enc(64) = 320
        fusion_in = embed_dim * 2 + lstm_hidden * 2 + 64

        self.fusion_fc1 = nn.Linear(fusion_in, 256)
        self.fusion_bn1 = nn.BatchNorm1d(256)
        self.fusion_drop1 = nn.Dropout(0.3)

        self.fusion_fc2 = nn.Linear(256, 128)
        self.fusion_residual = nn.Linear(fusion_in, 128)
        self.fusion_bn2 = nn.BatchNorm1d(128)
        self.fusion_drop2 = nn.Dropout(0.2)

        self.fusion_fc3 = nn.Linear(128, 64)
        self.fusion_drop3 = nn.Dropout(0.1)

        self.fusion_out = nn.Linear(64, 1)

    def _attend(self, lstm_out):
        """Apply attention over LSTM hidden states.

        Args:
            lstm_out: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size)
        """
        # scores: (batch, seq_len, 1)
        scores = self.attention(lstm_out)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)
        context = (weights * lstm_out).sum(dim=1)  # (batch, hidden_size)
        return context

    def forward(self, home_seq, away_seq, home_idx, away_idx, match_features):
        # Player embeddings
        home_emb = self.player_embed(home_idx)  # (batch, embed_dim)
        away_emb = self.player_embed(away_idx)

        # LSTM + attention for both players
        home_lstm_out, _ = self.sequence_lstm(home_seq)  # (batch, seq_len, hidden)
        away_lstm_out, _ = self.sequence_lstm(away_seq)

        home_attn = self._attend(home_lstm_out)  # (batch, hidden)
        away_attn = self._attend(away_lstm_out)

        # Match features
        match_enc = self.match_encoder(match_features)  # (batch, 64)

        # Fusion with residual
        combined = torch.cat([home_emb, away_emb, home_attn, away_attn, match_enc], dim=1)

        x = self.fusion_fc1(combined)
        x = self.fusion_bn1(x)
        x = torch.relu(x)
        x = self.fusion_drop1(x)

        x = self.fusion_fc2(x)
        residual = self.fusion_residual(combined)
        x = x + residual
        x = self.fusion_bn2(x)
        x = torch.relu(x)
        x = self.fusion_drop2(x)

        x = self.fusion_fc3(x)
        x = torch.relu(x)
        x = self.fusion_drop3(x)

        x = self.fusion_out(x)
        x = torch.sigmoid(x)
        return x


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def _build_player_sequences(store, player_to_idx, meta_list, labels,
                            mapping=None):
    """
    Build LSTM input sequences for every sample.

    For each match we need the home and away player's last SEQUENCE_LENGTH
    matches *before* the match date.

    Per-step features (10):
        0: won (0/1)
        1: my_prob (implied probability from form/elo, use weighted form as proxy)
        2: was_fav (0/1)
        3: margin (win rate momentum proxy)
        4: momentum (last5 - prev5 win rate)
        5: ace_rate
        6: first_serve_won
        7: bp_save_rate
        8: days_rest (normalised)
        9: surface_match (1 if same surface as current match, else 0)

    Returns:
        home_seqs: np.array  (N, SEQUENCE_LENGTH, 10)
        away_seqs: np.array  (N, SEQUENCE_LENGTH, 10)
        home_idxs: np.array  (N,)
        away_idxs: np.array  (N,)
    """
    records = store["records"]
    form_history = store.get("form_history", {})
    serve_history = store.get("serve_history", {})

    # Build a mapping from (winner_id, loser_id, date) -> record index for
    # fast lookups is not needed; we iterate meta and match to records.
    # Instead, build per-player chronological history from records.

    # --- Build per-player match history from records ---
    # Each entry: (date, won, surface, ace_rate, first_serve_won, bp_save_rate)
    player_history = {}
    for rec in records:
        w_id = rec["winner_id"]
        l_id = rec["loser_id"]
        date = rec["date"]
        surface = rec["surface"]

        w_serve = rec["winner_stats"].get("serve", {})
        l_serve = rec["loser_stats"].get("serve", {})

        w_fatigue = rec["winner_stats"].get("fatigue", {})
        l_fatigue = rec["loser_stats"].get("fatigue", {})

        if w_id not in player_history:
            player_history[w_id] = []
        player_history[w_id].append({
            "date": date,
            "won": 1.0,
            "surface": surface,
            "ace_rate": w_serve.get("ace_rate", 0.05),
            "first_serve_won": w_serve.get("first_serve_won", 0.70),
            "bp_save_rate": w_serve.get("bp_save_rate", 0.60),
            "days_rest": w_fatigue.get("days_since_last", 14),
        })

        if l_id not in player_history:
            player_history[l_id] = []
        player_history[l_id].append({
            "date": date,
            "won": 0.0,
            "surface": surface,
            "ace_rate": l_serve.get("ace_rate", 0.05),
            "first_serve_won": l_serve.get("first_serve_won", 0.70),
            "bp_save_rate": l_serve.get("bp_save_rate", 0.60),
            "days_rest": l_fatigue.get("days_since_last", 14),
        })

    # Also incorporate form_history for players not in records
    # (form_history entries are tuples: (date, won, surface, was_upset))
    for pid, hist in form_history.items():
        if pid not in player_history:
            player_history[pid] = []
            sh = serve_history.get(pid, [])
            for i, entry in enumerate(hist):
                s_entry = sh[i] if i < len(sh) else {}
                player_history[pid].append({
                    "date": entry[0],
                    "won": float(entry[1]),
                    "surface": entry[2],
                    "ace_rate": s_entry.get("ace_rate", 0.05),
                    "first_serve_won": s_entry.get("first_serve_won", 0.70),
                    "bp_save_rate": s_entry.get("bp_save_rate", 0.60),
                    "days_rest": 14,
                })

    # Sort each player's history by date
    for pid in player_history:
        player_history[pid].sort(key=lambda x: x["date"])

    # --- Resolve player IDs for each sample ---
    # Historical meta has: source=historical, winner, loser, date, league, surface
    # API meta has: source=api, home, away, date, league, surface, event_id
    #
    # For historical: we need winner_id, loser_id from records.
    # Build a lookup: (date, winner_name, loser_name) -> (winner_id, loser_id)
    # But meta doesn't always have names that match perfectly.
    # Alternative: for historical, index by position in records list.

    # Build record index by date for historical lookups
    records_by_date = {}
    for rec in records:
        d = rec["date"]
        if d not in records_by_date:
            records_by_date[d] = []
        records_by_date[d].append(rec)

    N = len(meta_list)
    home_seqs = np.zeros((N, SEQUENCE_LENGTH, LSTM_INPUT_SIZE), dtype=np.float32)
    away_seqs = np.zeros((N, SEQUENCE_LENGTH, LSTM_INPUT_SIZE), dtype=np.float32)
    home_idxs = np.zeros(N, dtype=np.int64)
    away_idxs = np.zeros(N, dtype=np.int64)

    # Track how many we resolve
    resolved = 0
    unresolved = 0

    for i, m in enumerate(meta_list):
        date = m["date"]
        source = m.get("source", "historical")
        match_surface = m.get("surface", "Hard")

        home_pid = None
        away_pid = None

        if source == "historical":
            # Try to find matching record
            winner_name = m.get("winner", "")
            loser_name = m.get("loser", "")
            label = labels[i]

            candidates = records_by_date.get(date, [])
            for rec in candidates:
                if (rec.get("winner_name", "") == winner_name and
                        rec.get("loser_name", "") == loser_name):
                    # label==1 means winner is home, label==0 means winner is away
                    if label > 0.5:
                        home_pid = rec["winner_id"]
                        away_pid = rec["loser_id"]
                    else:
                        home_pid = rec["loser_id"]
                        away_pid = rec["winner_id"]
                    break

        elif source == "api":
            # Use player_resolver to get sackmann IDs from API team IDs
            if mapping is None:
                mapping = {}
            home_name = m.get("home", "")
            away_name = m.get("away", "")
            event_id = m.get("event_id", "")

            # The meta doesn't store teamIDs directly. We need to find them
            # from the event. For API matches, the meta stores home/away names.
            # Try to resolve by searching the mapping for matching names.
            # The mapping is teamID -> {sackmann_id, method, name}.
            # Build reverse: name -> sackmann_id
            if not hasattr(_build_player_sequences, '_name_to_sackmann'):
                _build_player_sequences._name_to_sackmann = {}
                for tid, info in mapping.items():
                    name = info.get("name", "")
                    sid = info.get("sackmann_id", "")
                    if name and sid:
                        _build_player_sequences._name_to_sackmann[name.lower()] = sid
                    # Also try from teamID format: FIRST_LAST_LEAGUE
                    parts = tid.rsplit("_", 1)
                    if len(parts) == 2:
                        name_part = parts[0].replace("_", " ").title()
                        _build_player_sequences._name_to_sackmann[name_part.lower()] = sid

            name_map = _build_player_sequences._name_to_sackmann
            home_pid = name_map.get(home_name.lower())
            away_pid = name_map.get(away_name.lower())

        # Set player indices
        if home_pid:
            home_idxs[i] = player_to_idx.get(home_pid, 0)
        if away_pid:
            away_idxs[i] = player_to_idx.get(away_pid, 0)

        if home_pid or away_pid:
            resolved += 1
        else:
            unresolved += 1

        # Build sequences
        for pid, seqs_arr, slot in [(home_pid, home_seqs, i),
                                     (away_pid, away_seqs, i)]:
            if not pid or pid not in player_history:
                continue

            history = player_history[pid]
            # Get matches before this date
            recent = [h for h in history if h["date"] < date]
            recent = recent[-SEQUENCE_LENGTH:]

            for j, h in enumerate(recent):
                offset = SEQUENCE_LENGTH - len(recent) + j

                seqs_arr[slot, offset, 0] = h["won"]

                # my_prob: use rolling win rate as proxy
                prev_matches = recent[max(0, j - 10):j]
                if prev_matches:
                    seqs_arr[slot, offset, 1] = sum(
                        p["won"] for p in prev_matches
                    ) / len(prev_matches)
                else:
                    seqs_arr[slot, offset, 1] = 0.5

                # was_fav: was their recent form > 0.5
                seqs_arr[slot, offset, 2] = 1.0 if seqs_arr[slot, offset, 1] > 0.5 else 0.0

                # margin: rolling win rate - 0.5
                seqs_arr[slot, offset, 3] = seqs_arr[slot, offset, 1] - 0.5

                # momentum: last5 - prev5
                if j >= 10:
                    last5 = sum(recent[j - 5:j][k]["won"] for k in range(min(5, j))) / 5
                    prev5_slice = recent[max(0, j - 10):j - 5]
                    prev5 = sum(p["won"] for p in prev5_slice) / max(len(prev5_slice), 1)
                    seqs_arr[slot, offset, 4] = last5 - prev5
                elif j >= 5:
                    last5 = sum(recent[j - 5:j][k]["won"] for k in range(min(5, j))) / 5
                    seqs_arr[slot, offset, 4] = last5 - 0.5
                else:
                    seqs_arr[slot, offset, 4] = 0.0

                # ace_rate
                seqs_arr[slot, offset, 5] = h["ace_rate"]

                # first_serve_won
                seqs_arr[slot, offset, 6] = h["first_serve_won"]

                # bp_save_rate
                seqs_arr[slot, offset, 7] = h["bp_save_rate"]

                # days_rest (normalised to [0, 1])
                seqs_arr[slot, offset, 8] = min(h["days_rest"] / 30.0, 1.0)

                # surface_match
                seqs_arr[slot, offset, 9] = 1.0 if h["surface"] == match_surface else 0.0

    print(f"  Sequences built: {resolved} resolved, {unresolved} unresolved")

    # Clear cached name map
    if hasattr(_build_player_sequences, '_name_to_sackmann'):
        del _build_player_sequences._name_to_sackmann

    return home_seqs, away_seqs, home_idxs, away_idxs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    """Train the Mega Deep LSTM model."""
    print("=" * 60)
    print("MEGA DEEP LSTM TRAINER")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading feature store...")
    store_path = os.path.join(MODEL_DIR, "feature_store.pkl")
    with open(store_path, "rb") as f:
        store = pickle.load(f)
    print(f"  Records: {len(store['records'])}")

    print("Loading training data...")
    train_X = np.load(os.path.join(MODEL_DIR, "train_X.npy"))
    train_y = np.load(os.path.join(MODEL_DIR, "train_y.npy"))
    test_X = np.load(os.path.join(MODEL_DIR, "test_X.npy"))
    test_y = np.load(os.path.join(MODEL_DIR, "test_y.npy"))
    print(f"  Train: {train_X.shape}, Test: {test_X.shape}")

    print("Loading meta...")
    with open(os.path.join(MODEL_DIR, "train_meta.json")) as f:
        train_meta = json.load(f)
    with open(os.path.join(MODEL_DIR, "test_meta.json")) as f:
        test_meta = json.load(f)
    print(f"  Train meta: {len(train_meta)}, Test meta: {len(test_meta)}")

    # ------------------------------------------------------------------
    # Build player-to-index mapping from all unique player IDs in records
    # ------------------------------------------------------------------
    print("\nBuilding player index...")
    all_player_ids = set()
    for rec in store["records"]:
        all_player_ids.add(rec["winner_id"])
        all_player_ids.add(rec["loser_id"])
    # Also include any players from form/serve history
    for pid in store.get("form_history", {}):
        all_player_ids.add(pid)
    for pid in store.get("serve_history", {}):
        all_player_ids.add(pid)

    all_player_ids.discard("")
    sorted_ids = sorted(all_player_ids)
    player_to_idx = {pid: i + 1 for i, pid in enumerate(sorted_ids)}
    num_players = len(player_to_idx)
    print(f"  Unique players: {num_players}")

    # ------------------------------------------------------------------
    # Load player resolver mapping for API matches
    # ------------------------------------------------------------------
    print("Loading player resolver mapping...")
    try:
        from player_resolver import load_mapping
        api_mapping = load_mapping()
        print(f"  API mapping: {len(api_mapping)} entries")
    except Exception as e:
        print(f"  Could not load mapping: {e}")
        api_mapping = {}

    # ------------------------------------------------------------------
    # Build player sequences
    # ------------------------------------------------------------------
    print("\nBuilding player sequences for training data...")
    tr_home_seq, tr_away_seq, tr_home_idx, tr_away_idx = _build_player_sequences(
        store, player_to_idx, train_meta, train_y, mapping=api_mapping
    )

    print("Building player sequences for test data...")
    te_home_seq, te_away_seq, te_home_idx, te_away_idx = _build_player_sequences(
        store, player_to_idx, test_meta, test_y, mapping=api_mapping
    )

    # ------------------------------------------------------------------
    # Normalise match features
    # ------------------------------------------------------------------
    print("\nNormalising match features...")
    feat_mean = train_X.mean(axis=0)
    feat_std = train_X.std(axis=0)
    feat_std[feat_std == 0] = 1.0

    train_X_norm = (train_X - feat_mean) / feat_std
    test_X_norm = (test_X - feat_mean) / feat_std

    # Replace any remaining NaNs
    train_X_norm = np.nan_to_num(train_X_norm, 0.0)
    test_X_norm = np.nan_to_num(test_X_norm, 0.0)

    # ------------------------------------------------------------------
    # Build DataLoaders
    # ------------------------------------------------------------------
    print("Building data loaders...")

    train_dataset = TensorDataset(
        torch.FloatTensor(tr_home_seq),
        torch.FloatTensor(tr_away_seq),
        torch.LongTensor(tr_home_idx),
        torch.LongTensor(tr_away_idx),
        torch.FloatTensor(train_X_norm),
        torch.FloatTensor(train_y),
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(te_home_seq),
        torch.FloatTensor(te_away_seq),
        torch.LongTensor(te_home_idx),
        torch.LongTensor(te_away_idx),
        torch.FloatTensor(test_X_norm),
        torch.FloatTensor(test_y),
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # ------------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = MegaDeepNet(
        num_players=num_players,
        embed_dim=64,
        lstm_hidden=64,
        match_features=NUM_MATCH_FEATURES,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"  Embeddings:     {sum(p.numel() for p in model.player_embed.parameters()):,}")
    print(f"  LSTM:           {sum(p.numel() for p in model.sequence_lstm.parameters()):,}")
    print(f"  Attention:      {sum(p.numel() for p in model.attention.parameters()):,}")
    print(f"  Match encoder:  {sum(p.numel() for p in model.match_encoder.parameters()):,}")
    fusion_params = (
        sum(p.numel() for p in [model.fusion_fc1.weight, model.fusion_fc1.bias,
                                 model.fusion_bn1.weight, model.fusion_bn1.bias,
                                 model.fusion_fc2.weight, model.fusion_fc2.bias,
                                 model.fusion_residual.weight, model.fusion_residual.bias,
                                 model.fusion_bn2.weight, model.fusion_bn2.bias,
                                 model.fusion_fc3.weight, model.fusion_fc3.bias,
                                 model.fusion_out.weight, model.fusion_out.bias])
    )
    print(f"  Fusion:         {fusion_params:,}")

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = nn.BCELoss()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("\nTraining...")
    best_acc = 0.0
    best_state = None
    patience_counter = 0
    patience_limit = 20

    for epoch in range(200):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            h_seq, a_seq, h_idx, a_idx, feats, labs = [b.to(device) for b in batch]

            optimizer.zero_grad()
            pred = model(h_seq, a_seq, h_idx, a_idx, feats).squeeze()
            loss = criterion(pred, labs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in test_loader:
                    h_seq, a_seq, h_idx, a_idx, feats, labs = [b.to(device) for b in batch]
                    pred = model(h_seq, a_seq, h_idx, a_idx, feats).squeeze()
                    predicted = (pred > 0.5).float()
                    correct += (predicted == labs).sum().item()
                    total += labs.size(0)

            acc = correct / total if total > 0 else 0
            avg_loss = total_loss / n_batches
            lr = scheduler.get_last_lr()[0]

            print(f"  Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f} | "
                  f"Test acc: {acc * 100:.1f}% ({correct}/{total}) | "
                  f"LR: {lr:.6f}")

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"  Early stopping at epoch {epoch + 1} (patience={patience_limit})")
                    break

    # ------------------------------------------------------------------
    # Load best and final evaluation
    # ------------------------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    print(f"\n{'=' * 60}")
    print("FINAL EVALUATION")
    print(f"{'=' * 60}")

    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            h_seq, a_seq, h_idx, a_idx, feats, labs = [b.to(device) for b in batch]
            pred = model(h_seq, a_seq, h_idx, a_idx, feats).squeeze()
            predicted = (pred > 0.5).float()
            correct += (predicted == labs).sum().item()
            total += labs.size(0)
            all_probs.extend(pred.cpu().numpy().tolist())
            all_labels.extend(labs.cpu().numpy().tolist())

    acc = correct / total * 100 if total else 0
    print(f"Test accuracy: {acc:.1f}% ({correct}/{total})")
    print(f"Best accuracy: {best_acc * 100:.1f}%")

    # Calibration
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    winner_probs = np.maximum(probs, 1 - probs)
    predicted_correct = ((probs > 0.5) == (labels > 0.5))

    print(f"\nCalibration:")
    for name, lo, hi in [("50-60", 0.5, 0.6), ("60-70", 0.6, 0.7),
                          ("70-80", 0.7, 0.8), ("80-90", 0.8, 0.9),
                          ("90-100", 0.9, 1.01)]:
        mask = (winner_probs >= lo) & (winner_probs < hi)
        if mask.sum() > 0:
            actual = predicted_correct[mask].mean() * 100
            pred_avg = winner_probs[mask].mean() * 100
            print(f"  {name}%: pred {pred_avg:.1f}% -> actual {actual:.1f}% "
                  f"({mask.sum()} matches)")

    # ------------------------------------------------------------------
    # Save model and params
    # ------------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save test predictions for ensemble
    test_preds_path = os.path.join(MODEL_DIR, "mega_deep_test_preds.npy")
    np.save(test_preds_path, probs)
    print(f"Saved test predictions to {test_preds_path}")

    model_path = os.path.join(MODEL_DIR, "mega_deep.pt")
    torch.save(model.state_dict(), model_path)

    params_path = os.path.join(MODEL_DIR, "mega_deep_params.pkl")
    with open(params_path, "wb") as f:
        pickle.dump({
            "player_to_idx": player_to_idx,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "num_players": num_players,
            "embed_dim": 64,
            "lstm_hidden": 64,
            "match_features": NUM_MATCH_FEATURES,
            "sequence_length": SEQUENCE_LENGTH,
            "lstm_input_size": LSTM_INPUT_SIZE,
            "best_acc": best_acc,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nModel saved to {model_path}")
    print(f"Params saved to {params_path}")
    print(f"  Players: {num_players}")
    print(f"  Best test accuracy: {best_acc * 100:.1f}%")

    return model


if __name__ == "__main__":
    train()
