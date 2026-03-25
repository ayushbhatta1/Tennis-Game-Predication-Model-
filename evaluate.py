"""
Evaluation — all models on same test data with comprehensive metrics.

Metrics: accuracy, log loss, calibration, ROI simulation,
per-surface, per-league, upset detection.
"""

import json
import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")


def load_test_data():
    X_test = np.load(os.path.join(MODEL_DIR, "test_X.npy"))
    y_test = np.load(os.path.join(MODEL_DIR, "test_y.npy"))
    with open(os.path.join(MODEL_DIR, "test_meta.json")) as f:
        test_meta = json.load(f)
    with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)
    return X_test, y_test, test_meta, feature_names


def get_model_predictions(X_test, test_meta):
    """Get predictions from all available models."""
    predictions = {}

    # XGBoost
    xgb_path = os.path.join(MODEL_DIR, "mega_xgb.pkl")
    if os.path.exists(xgb_path):
        with open(xgb_path, "rb") as f:
            xgb_data = pickle.load(f)
        for variant in ["full", "history", "odds"]:
            model = xgb_data.get(variant)
            if model:
                if variant == "history":
                    X_sel = X_test[:, 18:]
                elif variant == "odds":
                    X_sel = np.hstack([X_test[:, :18], X_test[:, 64:]])
                else:
                    X_sel = X_test
                predictions[f"xgb_{variant}"] = model.predict_proba(X_sel)[:, 1]

    # Neural Network
    nn_params_path = os.path.join(MODEL_DIR, "mega_nn_params.pkl")
    if os.path.exists(nn_params_path):
        try:
            import torch
            import torch.nn as nn

            with open(nn_params_path, "rb") as f:
                params = pickle.load(f)

            mean, std = params["mean"], params["std"]
            input_size = params["input_size"]
            X_norm = (X_test - mean) / std

            class MegaNN(nn.Module):
                def __init__(self, sz):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(sz, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(32, 1), nn.Sigmoid(),
                    )
                def forward(self, x):
                    return self.net(x)

            all_states = params.get("all_states", [])
            if all_states:
                all_preds = []
                for state in all_states:
                    model = MegaNN(input_size)
                    model.load_state_dict(state)
                    model.eval()
                    with torch.no_grad():
                        preds = model(torch.FloatTensor(X_norm)).numpy().flatten()
                        all_preds.append(preds)
                predictions["nn_full"] = np.mean(all_preds, axis=0)
            else:
                pt_path = os.path.join(MODEL_DIR, "mega_nn.pt")
                if os.path.exists(pt_path):
                    model = MegaNN(input_size)
                    model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
                    model.eval()
                    with torch.no_grad():
                        predictions["nn_full"] = model(torch.FloatTensor(X_norm)).numpy().flatten()
        except Exception as e:
            print(f"  NN load error: {e}")

    # Deep LSTM
    deep_pred_path = os.path.join(MODEL_DIR, "mega_deep_test_preds.npy")
    if os.path.exists(deep_pred_path):
        preds = np.load(deep_pred_path)
        if len(preds) == len(X_test):
            predictions["deep_lstm"] = preds

    # CatBoost / LightGBM
    gbm_path = os.path.join(MODEL_DIR, "mega_gbm.pkl")
    CAT_COLS_ABS = {17, 44, 50, 51, 52}
    if os.path.exists(gbm_path):
        with open(gbm_path, "rb") as f:
            gbm_data = pickle.load(f)
        for model_type in ["catboost", "lgbm"]:
            for variant in ["full", "history", "odds"]:
                key = f"{model_type}_{variant}"
                model = gbm_data.get(key)
                if model:
                    if variant == "history":
                        indices = list(range(18, 66))
                    elif variant == "odds":
                        indices = list(range(18)) + [64, 65]
                    else:
                        indices = list(range(66))
                    X_sel = X_test[:, indices]
                    try:
                        predictions[key] = model.predict_proba(X_sel)[:, 1]
                    except Exception as e:
                        print(f"  {key} error: {e}")

    # Ensemble
    ens_path = os.path.join(MODEL_DIR, "mega_ensemble.pkl")
    if os.path.exists(ens_path):
        with open(ens_path, "rb") as f:
            ens = pickle.load(f)
        # Reconstruct stacking input
        model_names = ens["model_names"]
        stack_cols = []
        for name in model_names:
            if name in predictions:
                stack_cols.append(predictions[name])
            else:
                stack_cols.append(np.full(len(X_test), 0.5))
        if stack_cols:
            test_stack = np.column_stack(stack_cols)
            if ens.get("use_stacking") and ens.get("stacker"):
                predictions["ensemble_stack"] = ens["stacker"].predict_proba(test_stack)[:, 1]
            weights = ens.get("weights")
            if weights is not None:
                predictions["ensemble_weighted"] = np.average(test_stack, axis=1, weights=weights)

            # History-only ensemble
            hist_idx = ens.get("hist_model_indices", [])
            if hist_idx and ens.get("hist_stacker"):
                hist_stack = test_stack[:, hist_idx]
                predictions["ensemble_history"] = ens["hist_stacker"].predict_proba(hist_stack)[:, 1]

    return predictions


def compute_calibration(y_true, y_pred):
    """Compute calibration buckets."""
    winner_probs = np.maximum(y_pred, 1 - y_pred)
    correct = ((y_pred > 0.5) == y_true)

    cal = {}
    for name, lo, hi in [("50-60", 0.5, 0.6), ("60-70", 0.6, 0.7),
                          ("70-80", 0.7, 0.8), ("80-90", 0.8, 0.9),
                          ("90-100", 0.9, 1.01)]:
        mask = (winner_probs >= lo) & (winner_probs < hi)
        if mask.sum() > 0:
            cal[name] = {
                "total": int(mask.sum()),
                "correct": int(correct[mask].sum()),
                "actual_pct": round(correct[mask].mean() * 100, 1),
                "avg_predicted_pct": round(winner_probs[mask].mean() * 100, 1),
            }
        else:
            cal[name] = {"total": 0, "correct": 0, "actual_pct": 0, "avg_predicted_pct": 0}
    return cal


def evaluate():
    """Run full evaluation on all models."""
    print("Loading test data...")
    X_test, y_test, test_meta, feature_names = load_test_data()
    print(f"Test set: {len(X_test)} matches")

    print("\nLoading model predictions...")
    predictions = get_model_predictions(X_test, test_meta)
    print(f"Models found: {list(predictions.keys())}")

    if not predictions:
        print("No models available!")
        return

    # === Overall Comparison ===
    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON — {len(X_test)} March 2026 Matches")
    print(f"{'='*70}")
    print(f"  {'Model':25s} {'Accuracy':>10s} {'Log Loss':>10s} {'Correct':>10s}")
    print(f"  {'-'*55}")

    results = {}
    for name, preds in sorted(predictions.items()):
        preds_clipped = np.clip(preds, 0.01, 0.99)
        acc = accuracy_score(y_test, preds > 0.5)
        ll = log_loss(y_test, preds_clipped)
        correct = int(((preds > 0.5) == y_test).sum())
        print(f"  {name:25s} {acc*100:>9.1f}% {ll:>10.4f} {correct:>7d}/{len(y_test)}")
        results[name] = {
            "accuracy": round(acc * 100, 1),
            "log_loss": round(ll, 4),
            "correct": correct,
            "total": len(y_test),
        }

    # === Calibration for best model ===
    best_model = max(results, key=lambda k: results[k]["accuracy"])
    best_preds = predictions[best_model]
    cal = compute_calibration(y_test, best_preds)

    print(f"\n  Calibration ({best_model}):")
    for bucket, data in cal.items():
        if data["total"] > 0:
            print(f"    {bucket}%: pred {data['avg_predicted_pct']}% -> "
                  f"actual {data['actual_pct']}% ({data['total']})")

    # === Per-surface breakdown ===
    print(f"\n  Per-Surface ({best_model}):")
    surfaces = set(m.get("surface", "Unknown") for m in test_meta)
    surface_stats = {}
    for surface in sorted(surfaces):
        mask = np.array([m.get("surface", "") == surface for m in test_meta])
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], best_preds[mask] > 0.5)
            print(f"    {surface:10s}: {acc*100:.1f}% ({mask.sum()} matches)")
            surface_stats[surface] = {"accuracy": round(acc * 100, 1), "total": int(mask.sum())}

    # === Per-league ===
    print(f"\n  Per-League ({best_model}):")
    league_stats = {}
    for league in ["ATP", "WTA"]:
        mask = np.array([m.get("league", "") == league for m in test_meta])
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], best_preds[mask] > 0.5)
            print(f"    {league}: {acc*100:.1f}% ({mask.sum()} matches)")
            league_stats[league] = {"accuracy": round(acc * 100, 1), "total": int(mask.sum())}

    # === Upset detection ===
    print(f"\n  Upset Detection ({best_model}):")
    # Upset = predicted winner had >70% prob but lost
    upset_mask = (np.maximum(best_preds, 1-best_preds) >= 0.7) & ((best_preds > 0.5) != y_test)
    confident_mask = np.maximum(best_preds, 1-best_preds) >= 0.7
    if confident_mask.sum() > 0:
        upset_rate = upset_mask.sum() / confident_mask.sum()
        print(f"    Confident predictions (>70%): {confident_mask.sum()}")
        print(f"    Upsets in confident: {upset_mask.sum()} ({upset_rate*100:.1f}%)")
        print(f"    Confident accuracy: {(1-upset_rate)*100:.1f}%")

    # === Per-source breakdown ===
    for source in ["historical", "api"]:
        mask = np.array([m.get("source", "") == source for m in test_meta])
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], best_preds[mask] > 0.5)
            print(f"\n  Source={source}: {acc*100:.1f}% ({mask.sum()} matches)")

    # === Save report ===
    report = {
        "test_size": len(X_test),
        "models": results,
        "best_model": best_model,
        "best_accuracy": results[best_model]["accuracy"],
        "calibration": cal,
        "surface_stats": surface_stats,
        "league_stats": league_stats,
    }

    report_path = os.path.join(MODEL_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    return report


if __name__ == "__main__":
    evaluate()
