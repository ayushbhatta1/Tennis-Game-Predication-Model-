"""
Mega XGBoost Trainer — grid-search three model variants on 66-feature data.

Variants:
  1. Full      — all 66 features (indices 0-65)
  2. History   — history features only (indices 18-65, skipping odds block)
  3. Odds      — odds + flags only (indices 0-17, 64-65)

Uses temporal validation (last ~10% of training data by date) with early
stopping on logloss, then retrains best params on the full train set.
"""

import json
import os
import pickle
import itertools
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")


# ---------------------------------------------------------------------------
# Feature-index definitions for each variant
# ---------------------------------------------------------------------------
ALL_INDICES = list(range(66))
HISTORY_INDICES = list(range(18, 66))           # 18-65: everything after odds
ODDS_INDICES = list(range(18)) + [64, 65]       # 0-17 odds + 64-65 flags


def _load_data():
    """Load .npy arrays, feature names, and training metadata."""
    train_X = np.load(os.path.join(MODEL_DIR, "train_X.npy"))
    train_y = np.load(os.path.join(MODEL_DIR, "train_y.npy"))
    test_X = np.load(os.path.join(MODEL_DIR, "test_X.npy"))
    test_y = np.load(os.path.join(MODEL_DIR, "test_y.npy"))

    with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)

    with open(os.path.join(MODEL_DIR, "train_meta.json")) as f:
        train_meta = json.load(f)

    return train_X, train_y, test_X, test_y, feature_names, train_meta


def _temporal_val_split(train_X, train_y, train_meta, val_frac=0.10):
    """
    Split training data so the last ~val_frac (by date order) is the
    validation set.  The metadata already comes sorted by date from
    build_training_data, but we sort explicitly to be safe.
    """
    dates = [m["date"] for m in train_meta]
    order = np.argsort(dates, kind="stable")

    split_idx = int(len(order) * (1.0 - val_frac))

    tr_idx = order[:split_idx]
    va_idx = order[split_idx:]

    return (train_X[tr_idx], train_y[tr_idx],
            train_X[va_idx], train_y[va_idx])


# ---------------------------------------------------------------------------
# Grid search + training
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "max_depth": [4, 5, 6, 7],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [300, 500, 800, 1200],
    "subsample": [0.7, 0.8, 0.9],
}


def _grid_search(X_tr, y_tr, X_va, y_va, variant_name):
    """
    Exhaustive grid search with early stopping on the temporal validation set.
    Returns (best_params, best_logloss).
    """
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*(PARAM_GRID[k] for k in keys)))
    total = len(combos)

    print(f"\n  [{variant_name}] Grid search over {total} combinations ...")

    best_ll = float("inf")
    best_params = None
    t0 = time.time()

    for idx, values in enumerate(combos, 1):
        params = dict(zip(keys, values))

        model = xgb.XGBClassifier(
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            subsample=params["subsample"],
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            early_stopping_rounds=30,
            verbosity=0,
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        probs = model.predict_proba(X_va)[:, 1]
        ll = log_loss(y_va, probs)

        if ll < best_ll:
            best_ll = ll
            best_params = params.copy()
            best_params["best_iteration"] = model.best_iteration

        if idx % 20 == 0 or idx == total:
            elapsed = time.time() - t0
            print(f"    {idx}/{total}  best logloss={best_ll:.5f}  "
                  f"({elapsed:.0f}s)")

    print(f"  [{variant_name}] Best params: {best_params}  "
          f"(val logloss={best_ll:.5f})")
    return best_params, best_ll


def _train_final(X_train, y_train, params):
    """Train a final model on the full training set with the best params."""
    n_est = params.get("best_iteration", params["n_estimators"])
    # Use at least the early-stopped iteration count, cap at grid value
    n_est = min(n_est + 50, params["n_estimators"])

    model = xgb.XGBClassifier(
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        n_estimators=n_est,
        subsample=params["subsample"],
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def _evaluate(model, X, y, label):
    """Print accuracy and logloss for a dataset."""
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, preds)
    ll = log_loss(y, probs)
    print(f"    {label:18s}  acc={acc*100:.2f}%  logloss={ll:.5f}  "
          f"(n={len(y)})")
    return acc, ll


def _print_feature_importance(model, feature_names, top_n=20):
    """Print top-N feature importances for the Full model."""
    fi = model.feature_importances_
    order = np.argsort(fi)[::-1]

    print(f"\n  Feature importance (top {top_n}):")
    for rank, idx in enumerate(order[:top_n], 1):
        name = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
        print(f"    {rank:2d}. {name:30s}  {fi[idx]:.4f}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def train():
    """Run full training pipeline: grid search, train, evaluate, save."""
    print("=" * 60)
    print("  MEGA XGB TRAINER")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\nLoading data ...")
    train_X, train_y, test_X, test_y, feature_names, train_meta = _load_data()
    print(f"  train: {train_X.shape}   test: {test_X.shape}")
    print(f"  features: {len(feature_names)}")

    # ------------------------------------------------------------------
    # 2. Compute normalisation params (mean / std) from training set
    # ------------------------------------------------------------------
    tr_mean = train_X.mean(axis=0)
    tr_std = train_X.std(axis=0)
    tr_std[tr_std == 0] = 1.0
    norm_params = {"mean": tr_mean, "std": tr_std}

    # ------------------------------------------------------------------
    # 3. Temporal validation split (last ~10 % by date)
    # ------------------------------------------------------------------
    X_tr, y_tr, X_va, y_va = _temporal_val_split(
        train_X, train_y, train_meta, val_frac=0.10
    )
    print(f"  temporal split -> train {X_tr.shape[0]}  val {X_va.shape[0]}")

    # ------------------------------------------------------------------
    # 4. Define variants
    # ------------------------------------------------------------------
    variants = {
        "full": ALL_INDICES,
        "history": HISTORY_INDICES,
        "odds": ODDS_INDICES,
    }

    models = {}
    results = {}

    for vname, cols in variants.items():
        print(f"\n{'='*60}")
        print(f"  VARIANT: {vname.upper()}  ({len(cols)} features)")
        print(f"{'='*60}")

        # Slice features for this variant
        vX_tr = X_tr[:, cols]
        vy_tr = y_tr
        vX_va = X_va[:, cols]
        vy_va = y_va
        vX_train = train_X[:, cols]
        vy_train = train_y
        vX_test = test_X[:, cols]
        vy_test = test_y

        # Grid search on temporal validation split
        best_params, val_ll = _grid_search(vX_tr, vy_tr, vX_va, vy_va, vname)

        # Retrain on full training set with best params
        print(f"\n  Training final {vname} model on full train set ...")
        model = _train_final(vX_train, vy_train, best_params)

        # Evaluate
        print()
        _evaluate(model, vX_train, vy_train, "train")
        test_acc, test_ll = _evaluate(model, vX_test, vy_test, "test")

        models[vname] = model
        results[vname] = {
            "params": best_params,
            "val_logloss": val_ll,
            "test_acc": test_acc,
            "test_logloss": test_ll,
        }

    # ------------------------------------------------------------------
    # 5. Feature importance for Full model
    # ------------------------------------------------------------------
    _print_feature_importance(models["full"], feature_names)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for vname in ["full", "history", "odds"]:
        r = results[vname]
        print(f"  {vname:10s}  test_acc={r['test_acc']*100:.2f}%  "
              f"test_logloss={r['test_logloss']:.5f}  "
              f"val_logloss={r['val_logloss']:.5f}")

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    save_path = os.path.join(MODEL_DIR, "mega_xgb.pkl")
    payload = {
        "full": models["full"],
        "history": models["history"],
        "odds": models["odds"],
        "norm_params": norm_params,
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  Saved to {save_path}")

    return models, results


if __name__ == "__main__":
    train()
