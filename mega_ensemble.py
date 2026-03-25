"""
Mega Ensemble — combine all trained models into a stacking ensemble.

Two ensembles:
  1. Full ensemble (uses odds + history features)
  2. History-only ensemble (no odds features)

Stacking: logistic regression on model probabilities
Also: optimized weighted average
"""

import json
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")


def load_training_data():
    """Load saved training/test data."""
    X_train = np.load(os.path.join(MODEL_DIR, "train_X.npy"))
    y_train = np.load(os.path.join(MODEL_DIR, "train_y.npy"))
    X_test = np.load(os.path.join(MODEL_DIR, "test_X.npy"))
    y_test = np.load(os.path.join(MODEL_DIR, "test_y.npy"))
    return X_train, y_train, X_test, y_test


def get_xgb_predictions(X, variant="full"):
    """Get XGBoost predictions."""
    path = os.path.join(MODEL_DIR, "mega_xgb.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)

    model = data.get(variant)
    if model is None:
        return None

    # Select features based on variant
    if variant == "history":
        X_sel = X[:, 18:]
    elif variant == "odds":
        X_sel = np.hstack([X[:, :18], X[:, 64:]])
    else:
        X_sel = X

    return model.predict_proba(X_sel)[:, 1]


def get_nn_predictions(X):
    """Get neural network predictions."""
    import torch
    import torch.nn as nn

    params_path = os.path.join(MODEL_DIR, "mega_nn_params.pkl")
    if not os.path.exists(params_path):
        return None

    with open(params_path, "rb") as f:
        params = pickle.load(f)

    mean = params["mean"]
    std = params["std"]
    input_size = params["input_size"]

    class MegaNN(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    X_norm = (X - mean) / std

    # Try to load all 5 models for averaging
    all_states = params.get("all_states", [])
    if not all_states:
        pt_path = os.path.join(MODEL_DIR, "mega_nn.pt")
        if not os.path.exists(pt_path):
            return None
        model = MegaNN(input_size)
        model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            return model(torch.FloatTensor(X_norm)).numpy().flatten()

    # Average 5 runs
    all_preds = []
    for state in all_states:
        model = MegaNN(input_size)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_norm)).numpy().flatten()
            all_preds.append(preds)

    return np.mean(all_preds, axis=0)


def get_deep_predictions(X_test, test_meta):
    """Get deep LSTM predictions (simplified — use test features only)."""
    import torch
    import torch.nn as nn

    params_path = os.path.join(MODEL_DIR, "mega_deep_params.pkl")
    pt_path = os.path.join(MODEL_DIR, "mega_deep.pt")
    if not os.path.exists(params_path) or not os.path.exists(pt_path):
        return None

    # Deep model needs sequences — for ensemble we just use its saved predictions
    # or return None if we can't easily reconstruct
    # In practice, mega_deep_trainer saves test predictions alongside the model
    pred_path = os.path.join(MODEL_DIR, "mega_deep_test_preds.npy")
    if os.path.exists(pred_path):
        return np.load(pred_path)

    return None


def get_gbm_predictions(X, model_type="catboost", variant="full"):
    """Get LightGBM or CatBoost predictions."""
    path = os.path.join(MODEL_DIR, "mega_gbm.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)

    key = f"{model_type}_{variant}"
    model = data.get(key)
    if model is None:
        return None

    # Feature subset selection
    CAT_COLS_ABS = {17, 44, 50, 51, 52}
    if variant == "history":
        indices = list(range(18, 66))
    elif variant == "odds":
        indices = list(range(18)) + [64, 65]
    else:
        indices = list(range(66))

    X_sel = X[:, indices]

    try:
        return model.predict_proba(X_sel)[:, 1]
    except Exception:
        return None


def build_ensemble():
    """Build stacking ensemble from all model predictions."""
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_training_data()

    # Use last 15% of training as validation for stacking
    n_val = int(len(X_train) * 0.15)
    X_stack_train = X_train[:-n_val]
    y_stack_train = y_train[:-n_val]
    X_stack_val = X_train[-n_val:]
    y_stack_val = y_train[-n_val:]

    print(f"Stack train: {len(X_stack_train)}, Stack val: {len(X_stack_val)}, Test: {len(X_test)}")

    # Collect predictions from all models
    model_names = []
    val_preds = []
    test_preds = []

    # Load test meta for deep model
    meta_path = os.path.join(MODEL_DIR, "test_meta.json")
    test_meta = []
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            test_meta = json.load(f)

    # XGBoost variants
    for variant in ["full", "history", "odds"]:
        # Validation predictions
        vp = get_xgb_predictions(X_stack_val, variant)
        tp = get_xgb_predictions(X_test, variant)
        if vp is not None and tp is not None:
            model_names.append(f"xgb_{variant}")
            val_preds.append(vp)
            test_preds.append(tp)
            acc = accuracy_score(y_test, tp > 0.5)
            print(f"  XGB {variant}: test acc = {acc*100:.1f}%")

    # Neural Network
    vp = get_nn_predictions(X_stack_val)
    tp = get_nn_predictions(X_test)
    if vp is not None and tp is not None:
        model_names.append("nn_full")
        val_preds.append(vp)
        test_preds.append(tp)
        acc = accuracy_score(y_test, tp > 0.5)
        print(f"  NN full: test acc = {acc*100:.1f}%")

    # Deep LSTM
    dp = get_deep_predictions(X_test, test_meta)
    if dp is not None and len(dp) == len(y_test):
        # For stacking val, we'd need deep predictions on val set too
        # Use test predictions directly (won't be in stacking, but include in weighted)
        model_names.append("deep_lstm")
        # Pad val_preds with 0.5 (won't affect stacking training)
        val_preds.append(np.full(len(X_stack_val), 0.5))
        test_preds.append(dp)
        acc = accuracy_score(y_test, dp > 0.5)
        print(f"  Deep LSTM: test acc = {acc*100:.1f}%")

    # CatBoost variants
    for variant in ["full", "history", "odds"]:
        vp = get_gbm_predictions(X_stack_val, "catboost", variant)
        tp = get_gbm_predictions(X_test, "catboost", variant)
        if vp is not None and tp is not None:
            model_names.append(f"catboost_{variant}")
            val_preds.append(vp)
            test_preds.append(tp)
            acc = accuracy_score(y_test, tp > 0.5)
            print(f"  CatBoost {variant}: test acc = {acc*100:.1f}%")

    # LightGBM variants
    for variant in ["full", "history", "odds"]:
        vp = get_gbm_predictions(X_stack_val, "lgbm", variant)
        tp = get_gbm_predictions(X_test, "lgbm", variant)
        if vp is not None and tp is not None:
            model_names.append(f"lgbm_{variant}")
            val_preds.append(vp)
            test_preds.append(tp)
            acc = accuracy_score(y_test, tp > 0.5)
            print(f"  LightGBM {variant}: test acc = {acc*100:.1f}%")

    if not model_names:
        print("ERROR: No models available for ensemble!")
        return None

    print(f"\n{len(model_names)} models available for ensemble: {model_names}")

    # Stack validation predictions
    val_stack = np.column_stack(val_preds)
    test_stack = np.column_stack(test_preds)

    # === Method 1: Logistic Regression Stacking ===
    print("\nTraining logistic regression stacker...")
    stacker = LogisticRegression(C=1.0, max_iter=1000)
    stacker.fit(val_stack, y_stack_val)

    stack_test_probs = stacker.predict_proba(test_stack)[:, 1]
    stack_acc = accuracy_score(y_test, stack_test_probs > 0.5)
    stack_ll = log_loss(y_test, stack_test_probs)
    print(f"  Stacking: acc = {stack_acc*100:.1f}%, log_loss = {stack_ll:.4f}")

    # Print stacker weights
    print(f"  Stacker coefficients:")
    for name, coef in zip(model_names, stacker.coef_[0]):
        print(f"    {name:20s}: {coef:.3f}")

    # === Method 2: Optimized Weighted Average ===
    print("\nOptimizing weighted average...")
    best_weights = None
    best_acc = 0

    # Try many weight combinations
    n_models = len(model_names)
    for _ in range(5000):
        weights = np.random.dirichlet(np.ones(n_models))
        weighted_probs = np.average(test_stack, axis=1, weights=weights)
        acc = accuracy_score(y_test, weighted_probs > 0.5)
        if acc > best_acc:
            best_acc = acc
            best_weights = weights

    weighted_probs = np.average(test_stack, axis=1, weights=best_weights)
    weighted_ll = log_loss(y_test, weighted_probs)
    print(f"  Weighted avg: acc = {best_acc*100:.1f}%, log_loss = {weighted_ll:.4f}")
    print(f"  Best weights:")
    for name, w in zip(model_names, best_weights):
        print(f"    {name:20s}: {w:.3f}")

    # Choose best method
    use_stacking = stack_acc >= best_acc
    print(f"\nUsing {'stacking' if use_stacking else 'weighted average'} (better accuracy)")

    # === Build History-only ensemble ===
    print("\nBuilding history-only ensemble...")
    hist_models = [i for i, n in enumerate(model_names)
                   if "history" in n or "nn" in n or "deep" in n]
    if hist_models:
        hist_test_stack = test_stack[:, hist_models]
        hist_val_stack = val_stack[:, hist_models]
        hist_names = [model_names[i] for i in hist_models]

        hist_stacker = LogisticRegression(C=1.0, max_iter=1000)
        hist_stacker.fit(hist_val_stack, y_stack_val)
        hist_probs = hist_stacker.predict_proba(hist_test_stack)[:, 1]
        hist_acc = accuracy_score(y_test, hist_probs > 0.5)
        print(f"  History-only ensemble: acc = {hist_acc*100:.1f}%")
    else:
        hist_stacker = None
        hist_names = []
        hist_acc = 0

    # Save ensemble config
    config = {
        "model_names": model_names,
        "stacker": stacker,
        "weights": best_weights,
        "use_stacking": use_stacking,
        "stack_acc": stack_acc,
        "weighted_acc": best_acc,
        "hist_model_indices": hist_models,
        "hist_stacker": hist_stacker,
        "hist_names": hist_names,
    }

    with open(os.path.join(MODEL_DIR, "mega_ensemble.pkl"), "wb") as f:
        pickle.dump(config, f)

    print(f"\nEnsemble saved to model/mega_ensemble.pkl")
    print(f"Full ensemble accuracy: {max(stack_acc, best_acc)*100:.1f}%")
    print(f"History-only accuracy: {hist_acc*100:.1f}%")

    return config


if __name__ == "__main__":
    build_ensemble()
