"""
Neural Network prediction interface for live matches.
Loads the trained model and predicts upcoming events.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn

from nn_model import extract_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

_model = None
_mean = None
_std = None


class TennisNet(nn.Module):
    def __init__(self, input_size=18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 48),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(24, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def load_nn_model():
    """Load the trained model (cached in memory)."""
    global _model, _mean, _std

    if _model is not None:
        return _model, _mean, _std

    pt_path = os.path.join(MODEL_DIR, "tennis_nn.pt")
    params_path = os.path.join(MODEL_DIR, "norm_params.pkl")

    if not os.path.exists(pt_path) or not os.path.exists(params_path):
        return None, None, None

    with open(params_path, "rb") as f:
        params = pickle.load(f)

    _mean = params["mean"]
    _std = params["std"]
    input_size = params["input_size"]

    _model = TennisNet(input_size)
    _model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    _model.eval()

    return _model, _mean, _std


def nn_predict_event(event):
    """
    Run NN prediction on a single event.
    Returns dict with home/away win probabilities, or None if can't predict.
    """
    model, mean, std = load_nn_model()
    if model is None:
        return None

    # Use opening odds for live/upcoming (they ARE the current pre-match odds)
    # For upcoming events, fairOdds IS the pre-match odds
    features = extract_features(event, use_opening=False)
    if features is None:
        return None

    features_norm = (features - mean) / std
    with torch.no_grad():
        raw_prob = model(torch.FloatTensor(features_norm).unsqueeze(0)).item()

    # Temperature scaling: soften extreme predictions
    # Convert to logit, scale, convert back
    eps = 1e-6
    logit = np.log((raw_prob + eps) / (1 - raw_prob + eps))
    temperature = 2.0  # >1 = softer probabilities
    scaled_logit = logit / temperature
    home_prob = 1 / (1 + np.exp(-scaled_logit))

    # Clip to reasonable range (no tennis match is 99%+ certain pre-match)
    home_prob = np.clip(home_prob, 0.05, 0.95)

    return {
        "nn_home_prob": round(home_prob * 100, 1),
        "nn_away_prob": round((1 - home_prob) * 100, 1),
        "nn_winner": "home" if home_prob > 0.5 else "away",
        "nn_confidence": round(abs(home_prob - 0.5) * 200, 1),
    }
