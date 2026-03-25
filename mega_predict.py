"""
Mega Predict — unified prediction interface for live matches.

Replaces nn_predict.py. Uses the full ensemble pipeline:
  1. Resolve player IDs (API -> Sackmann)
  2. Build 66-feature vector from feature store + live odds
  3. Run all available models
  4. Return ensemble prediction with temperature scaling + clipping
"""

import json
import os
import pickle

import numpy as np

from feature_engine import (
    build_feature_vector, extract_odds_features_from_event, NUM_FEATURES
)
from player_resolver import load_mapping, get_sackmann_id

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")

# When True, use XGBoost prediction directly (skip broken NN/CatBoost ensemble).
# Set to False to re-enable the full ensemble after models are retrained.
USE_XGB_ONLY = True

# Cached state
_models_loaded = False
_mapping = None
_store = None
_xgb_models = None
_nn_model = None
_nn_params = None
_gbm_models = None
_ensemble = None


def _load_models():
    """Load all models into memory (once)."""
    global _models_loaded, _mapping, _store
    global _xgb_models, _nn_model, _nn_params, _gbm_models, _ensemble

    if _models_loaded:
        return

    # Player mapping
    _mapping = load_mapping()

    # Feature store (for player stats lookups)
    store_path = os.path.join(MODEL_DIR, "feature_store.pkl")
    if os.path.exists(store_path):
        with open(store_path, "rb") as f:
            _store = pickle.load(f)

    # XGBoost
    xgb_path = os.path.join(MODEL_DIR, "mega_xgb.pkl")
    if os.path.exists(xgb_path):
        with open(xgb_path, "rb") as f:
            _xgb_models = pickle.load(f)

    # Neural Network
    nn_params_path = os.path.join(MODEL_DIR, "mega_nn_params.pkl")
    if os.path.exists(nn_params_path):
        try:
            import torch
            import torch.nn as nn

            with open(nn_params_path, "rb") as f:
                _nn_params = pickle.load(f)

            class MegaNN(nn.Module):
                def __init__(self, input_size):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_size, 128),
                        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(128, 64),
                        nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(64, 32),
                        nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(32, 1), nn.Sigmoid(),
                    )
                def forward(self, x):
                    return self.net(x)

            input_size = _nn_params["input_size"]
            all_states = _nn_params.get("all_states", [])
            if all_states:
                _nn_model = []
                for state in all_states:
                    m = MegaNN(input_size)
                    m.load_state_dict(state)
                    m.eval()
                    _nn_model.append(m)
            else:
                pt_path = os.path.join(MODEL_DIR, "mega_nn.pt")
                if os.path.exists(pt_path):
                    m = MegaNN(input_size)
                    m.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
                    m.eval()
                    _nn_model = [m]
        except Exception:
            pass

    # GBM
    gbm_path = os.path.join(MODEL_DIR, "mega_gbm.pkl")
    if os.path.exists(gbm_path):
        with open(gbm_path, "rb") as f:
            _gbm_models = pickle.load(f)

    # Ensemble
    ens_path = os.path.join(MODEL_DIR, "mega_ensemble.pkl")
    if os.path.exists(ens_path):
        with open(ens_path, "rb") as f:
            _ensemble = pickle.load(f)

    _models_loaded = True


def _get_player_stats(team_id, opponent_tid, date_str, surface):
    """Look up player stats from the feature store."""
    if not _store or not _mapping:
        return {}

    sid = get_sackmann_id(team_id, _mapping)
    opp_sid = get_sackmann_id(opponent_tid, _mapping)
    if not sid:
        return {}

    date_compact = date_str.replace("-", "")[:8]

    from build_training_data import _lookup_player_stats
    return _lookup_player_stats(_store, sid, opp_sid or "", date_compact, surface)


def predict_match(event):
    """
    Predict a single match using the full mega ensemble.

    Args:
        event: API event dict

    Returns:
        dict with prediction details, or None if can't predict
    """
    _load_models()

    # Get team info
    teams = event.get("teams", {})
    home_tid = teams.get("home", {}).get("teamID", "")
    away_tid = teams.get("away", {}).get("teamID", "")
    if not home_tid or not away_tid:
        return None

    # Get match date
    date_str = event.get("status", {}).get("startsAt", "")[:10]
    league = event.get("leagueID", "ATP")

    # Extract odds
    odds_features = extract_odds_features_from_event(event, use_opening=False)
    has_odds = odds_features is not None

    # Get player stats from feature store
    home_stats = _get_player_stats(home_tid, away_tid, date_str, "Hard")
    away_stats = _get_player_stats(away_tid, home_tid, date_str, "Hard")
    has_history = bool(home_stats.get("player_id"))

    # Build feature vector
    vec = build_feature_vector(
        odds_features=odds_features,
        home_stats=home_stats,
        away_stats=away_stats,
        surface="Hard",
        round_val="R32",
        tourney_level="A",
        best_of=3,
        league=league,
        has_odds=has_odds,
        has_history=has_history,
    )

    vec = np.nan_to_num(vec, 0.0)
    X = vec.reshape(1, -1)

    # Collect predictions from all models
    model_preds = {}

    # XGBoost
    if _xgb_models:
        model = _xgb_models.get("full")
        if model:
            model_preds["xgb"] = model.predict_proba(X)[0, 1]

    # Neural Network
    if _nn_model and _nn_params:
        import torch
        mean, std = _nn_params["mean"], _nn_params["std"]
        X_norm = (X - mean) / std
        preds = []
        for m in _nn_model:
            with torch.no_grad():
                p = m(torch.FloatTensor(X_norm)).item()
                preds.append(p)
        # Flip: mega NN was trained with inverted labels (9% accuracy -> 91% when flipped)
        model_preds["nn"] = 1.0 - np.mean(preds)

    # CatBoost — removed: model has label inversion and collapsed predictions (0/15 accuracy)

    # --- Final prediction ---
    # When USE_XGB_ONLY is True and XGBoost produced a prediction, use it
    # directly instead of routing through the ensemble stacker (which is
    # polluted by the currently-broken NN and CatBoost weights).
    if USE_XGB_ONLY and "xgb" in model_preds:
        home_prob = model_preds["xgb"]
    elif _ensemble and model_preds:
        # Full ensemble path (fallback when XGB unavailable or flag is off)
        model_names = _ensemble["model_names"]
        stack_cols = []
        for name in model_names:
            # Map ensemble model names to available predictions
            short = name.replace("_full", "")
            if short in model_preds:
                stack_cols.append(model_preds[short])
            elif name in model_preds:
                stack_cols.append(model_preds[name])
            else:
                stack_cols.append(0.5)

        test_stack = np.array(stack_cols).reshape(1, -1)

        if _ensemble.get("use_stacking") and _ensemble.get("stacker"):
            home_prob = _ensemble["stacker"].predict_proba(test_stack)[0, 1]
        elif _ensemble.get("weights") is not None:
            home_prob = np.average(stack_cols, weights=_ensemble["weights"])
        else:
            home_prob = np.mean(list(model_preds.values()))
    elif model_preds:
        home_prob = np.mean(list(model_preds.values()))
    else:
        return None

    # Temperature scaling
    eps = 1e-6
    logit = np.log((home_prob + eps) / (1 - home_prob + eps))
    temperature = 1.5
    scaled_logit = logit / temperature
    home_prob = 1 / (1 + np.exp(-scaled_logit))

    # Clip to reasonable range
    home_prob = float(np.clip(home_prob, 0.05, 0.95))

    return {
        "mega_home_prob": round(home_prob * 100, 1),
        "mega_away_prob": round((1 - home_prob) * 100, 1),
        "mega_winner": "home" if home_prob > 0.5 else "away",
        "mega_confidence": round(abs(home_prob - 0.5) * 200, 1),
        "mega_models_used": list(model_preds.keys()),
        "mega_has_history": has_history,
        "mega_has_odds": has_odds,
        "mega_individual": {k: round(v * 100, 1) for k, v in model_preds.items()},
    }


def is_available():
    """Check if mega models are available."""
    return (
        os.path.exists(os.path.join(MODEL_DIR, "mega_xgb.pkl")) or
        os.path.exists(os.path.join(MODEL_DIR, "mega_nn.pt")) or
        os.path.exists(os.path.join(MODEL_DIR, "mega_ensemble.pkl"))
    )
