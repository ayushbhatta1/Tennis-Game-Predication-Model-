"""
Mega GBM Trainer — trains LightGBM and CatBoost models on the unified
66-feature tennis dataset.

Three variants per model type:
  1. Full      — all 66 features
  2. History   — indices 18-65
  3. Odds      — indices 0-17 + 64-65

Save everything to model/mega_gbm.pkl.
"""

import itertools
import json
import os
import pickle
import time
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ---------------------------------------------------------------------------
# Feature-subset definitions
# ---------------------------------------------------------------------------
FULL_INDICES = list(range(66))
HISTORY_INDICES = list(range(18, 66))
ODDS_INDICES = list(range(0, 18)) + [64, 65]

VARIANT_SPECS = {
    "full": FULL_INDICES,
    "history": HISTORY_INDICES,
    "odds": ODDS_INDICES,
}

# Categorical feature *original* column indices (absolute in the 66-col space)
CAT_COLS_ABS = {17, 44, 50, 51, 52}


def _cat_indices_for_subset(subset_indices):
    """Map absolute categorical column positions into relative positions
    within a feature subset."""
    idx_map = {abs_idx: rel_idx for rel_idx, abs_idx in enumerate(subset_indices)}
    return sorted(idx_map[c] for c in CAT_COLS_ABS if c in idx_map)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data():
    """Load training / test arrays and feature names."""
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
    """Split training data into train / validation by temporal order.

    Uses dates from train_meta to keep the last *val_frac* of matches
    (by date) as a held-out validation set.
    """
    dates = [m.get("date", "") for m in train_meta]
    n = len(dates)
    sorted_idx = sorted(range(n), key=lambda i: dates[i])

    split_point = int(n * (1.0 - val_frac))
    train_idx = sorted_idx[:split_point]
    val_idx = sorted_idx[split_point:]

    return (
        train_X[train_idx], train_y[train_idx],
        train_X[val_idx], train_y[val_idx],
    )


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------

LGBM_PARAM_GRID = {
    "num_leaves": [31, 63, 127],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [300, 500, 800],
    "subsample": [0.7, 0.8],
}


def _lgbm_grid_keys():
    keys = sorted(LGBM_PARAM_GRID.keys())
    return keys, list(itertools.product(*(LGBM_PARAM_GRID[k] for k in keys)))


def _train_lgbm(train_X, train_y, val_X, val_y, cat_indices, feature_names_sub):
    """Grid-search LightGBM and return the best fitted model."""
    try:
        import lightgbm as lgb
    except (ImportError, OSError):
        print("[WARNING] lightgbm not available (libomp may be missing). "
              "Skipping LightGBM training.")
        return None

    keys, combos = _lgbm_grid_keys()
    best_model = None
    best_loss = float("inf")
    best_params = None
    total = len(combos)

    print(f"  LightGBM grid search: {total} combinations")

    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        model = lgb.LGBMClassifier(
            objective="binary",
            num_leaves=params["num_leaves"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            subsample=params["subsample"],
            subsample_freq=1,
            random_state=42,
            verbose=-1,
        )

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model.fit(
            train_X, train_y,
            eval_set=[(val_X, val_y)],
            eval_metric="logloss",
            categorical_feature=cat_indices if cat_indices else "auto",
            callbacks=callbacks,
        )

        proba = model.predict_proba(val_X)[:, 1]
        loss = log_loss(val_y, proba)

        if loss < best_loss:
            best_loss = loss
            best_model = model
            best_params = params

        if i % 10 == 0 or i == total:
            print(f"    [{i}/{total}] best val logloss so far: {best_loss:.5f}")

    print(f"  Best LightGBM params: {best_params}  (val logloss={best_loss:.5f})")
    return best_model


# ---------------------------------------------------------------------------
# CatBoost training
# ---------------------------------------------------------------------------

CB_PARAM_GRID = {
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "iterations": [300, 500, 800],
}


def _cb_grid_keys():
    keys = sorted(CB_PARAM_GRID.keys())
    return keys, list(itertools.product(*(CB_PARAM_GRID[k] for k in keys)))


def _train_catboost(train_X, train_y, val_X, val_y, cat_indices):
    """Grid-search CatBoost and return the best fitted model."""
    from catboost import CatBoostClassifier, Pool

    # Treat all features as numeric — binary features work fine numerically
    # CatBoost's native categorical handling causes issues with float arrays
    train_pool = Pool(train_X, label=train_y)
    val_pool = Pool(val_X, label=val_y)

    keys, combos = _cb_grid_keys()
    best_model = None
    best_loss = float("inf")
    best_params = None
    total = len(combos)

    print(f"  CatBoost grid search: {total} combinations")

    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        model = CatBoostClassifier(
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            iterations=params["iterations"],
            eval_metric="Logloss",
            random_seed=42,
            verbose=0,
            early_stopping_rounds=50,
        )

        model.fit(train_pool, eval_set=val_pool, verbose=0)

        proba = model.predict_proba(val_pool)[:, 1]
        loss = log_loss(val_y, proba)

        if loss < best_loss:
            best_loss = loss
            best_model = model
            best_params = params

        if i % 5 == 0 or i == total:
            print(f"    [{i}/{total}] best val logloss so far: {best_loss:.5f}")

    print(f"  Best CatBoost params: {best_params}  (val logloss={best_loss:.5f})")
    return best_model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(model, test_X, test_y, label, is_catboost=False, cat_indices=None):
    """Print accuracy and log-loss on the test set."""
    if model is None:
        print(f"  {label:25s}  -- skipped (model unavailable) --")
        return

    X_eval = test_X

    preds = model.predict(X_eval)
    proba = model.predict_proba(X_eval)[:, 1]
    acc = accuracy_score(test_y, preds)
    ll = log_loss(test_y, proba)
    print(f"  {label:25s}  accuracy={acc:.4f}  logloss={ll:.5f}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train():
    """Train all GBM variants and save to model/mega_gbm.pkl."""
    print("=" * 60)
    print("Mega GBM Trainer")
    print("=" * 60)

    # --- load data ---
    print("\nLoading data...")
    train_X, train_y, test_X, test_y, feature_names, train_meta = _load_data()
    print(f"  train: {train_X.shape}   test: {test_X.shape}")

    # --- temporal validation split ---
    print("Creating temporal validation split (last 10%)...")
    tr_X, tr_y, va_X, va_y = _temporal_val_split(train_X, train_y, train_meta)
    print(f"  fit:  {tr_X.shape}   val: {va_X.shape}")

    results = {}

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------
    lgbm_available = True
    try:
        import lightgbm  # noqa: F401
    except (ImportError, OSError):
        lgbm_available = False
        print("\n[WARNING] lightgbm is not available (libomp may be missing). "
              "All LightGBM variants will be set to None.")

    for variant, indices in VARIANT_SPECS.items():
        key = f"lgbm_{variant}"
        if not lgbm_available:
            results[key] = None
            continue

        cat_idx = _cat_indices_for_subset(indices)
        sub_names = [feature_names[i] for i in indices] if feature_names else None

        print(f"\n--- LightGBM [{variant}] ({len(indices)} features, "
              f"cat_indices={cat_idx}) ---")

        t0 = time.time()
        model = _train_lgbm(
            tr_X[:, indices], tr_y,
            va_X[:, indices], va_y,
            cat_idx, sub_names,
        )
        elapsed = time.time() - t0
        print(f"  Trained in {elapsed:.1f}s")
        results[key] = model

    # ------------------------------------------------------------------
    # CatBoost
    # ------------------------------------------------------------------
    for variant, indices in VARIANT_SPECS.items():
        key = f"catboost_{variant}"
        cat_idx = _cat_indices_for_subset(indices)

        print(f"\n--- CatBoost [{variant}] ({len(indices)} features, "
              f"cat_indices={cat_idx}) ---")

        t0 = time.time()
        model = _train_catboost(
            tr_X[:, indices], tr_y,
            va_X[:, indices], va_y,
            cat_idx,
        )
        elapsed = time.time() - t0
        print(f"  Trained in {elapsed:.1f}s")
        results[key] = model

    # ------------------------------------------------------------------
    # Evaluation on test set
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test-set evaluation")
    print("=" * 60)

    for variant, indices in VARIANT_SPECS.items():
        lgbm_key = f"lgbm_{variant}"
        cb_key = f"catboost_{variant}"
        cat_idx = _cat_indices_for_subset(indices)
        tX = test_X[:, indices]

        _evaluate(results[lgbm_key], tX, test_y, lgbm_key,
                  is_catboost=False)
        _evaluate(results[cb_key], tX, test_y, cb_key,
                  is_catboost=True, cat_indices=cat_idx)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = os.path.join(MODEL_DIR, "mega_gbm.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved {len(results)} models to {out_path}")
    print("Keys:", sorted(results.keys()))

    return results


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train()
