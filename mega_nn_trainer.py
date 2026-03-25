"""
Mega Neural Network Trainer — 5-run ensemble with cosine annealing.
Architecture: 66 -> 128(BN,ReLU,Drop0.3) -> 64(BN,ReLU,Drop0.2) -> 32(BN,ReLU,Drop0.1) -> 1(Sigmoid)
"""

import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, log_loss
from pathlib import Path


class MegaNN(nn.Module):
    def __init__(self, input_size=66):
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
        return self.net(x).squeeze(-1)


def train():
    model_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "model"

    # ---- load data ----
    train_X = np.load(model_dir / "train_X.npy")
    train_y = np.load(model_dir / "train_y.npy")
    test_X = np.load(model_dir / "test_X.npy")
    test_y = np.load(model_dir / "test_y.npy")

    input_size = train_X.shape[1]
    print(f"Training data: {train_X.shape[0]} samples, {input_size} features")
    print(f"Test data:     {test_X.shape[0]} samples")

    # ---- NaN handling ----
    train_X = np.nan_to_num(train_X, 0.0).astype(np.float32)
    test_X = np.nan_to_num(test_X, 0.0).astype(np.float32)

    # ---- feature normalization ----
    # Compute mean/std from non-zero values for odds features (indices 0-17)
    # to handle the mixed zero/non-zero distribution
    feat_mean = train_X.mean(axis=0).astype(np.float32)
    feat_std = train_X.std(axis=0).astype(np.float32)
    feat_std[feat_std < 1e-8] = 1.0

    train_X = ((train_X - feat_mean) / feat_std).astype(np.float32)
    test_X = ((test_X - feat_mean) / feat_std).astype(np.float32)

    # ---- validation split: last 10 % of training data (temporal order) ----
    n = len(train_X)
    split = int(n * 0.9)
    val_X, val_y = train_X[split:], train_y[split:]
    trn_X, trn_y = train_X[:split], train_y[:split]

    print(f"Train split:   {trn_X.shape[0]} samples")
    print(f"Val split:     {val_X.shape[0]} samples")

    # tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}\n")

    trn_ds = TensorDataset(torch.tensor(trn_X), torch.tensor(trn_y, dtype=torch.float32))
    val_Xt = torch.tensor(val_X).to(device)
    val_yt = torch.tensor(val_y, dtype=torch.float32).to(device)
    test_Xt = torch.tensor(test_X).to(device)

    trn_loader = DataLoader(trn_ds, batch_size=512, shuffle=True)

    # ---- 5-run ensemble ----
    num_runs = 5
    all_state_dicts = []
    best_single_state = None
    best_single_val_loss = float("inf")
    all_test_preds = []

    for run in range(1, num_runs + 1):
        print(f"{'='*50}")
        print(f"Run {run}/{num_runs}")
        print(f"{'='*50}")

        model = MegaNN(input_size=input_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=300, eta_min=1e-6
        )
        criterion = nn.BCELoss()

        best_val_loss = float("inf")
        best_epoch_state = None
        patience_counter = 0
        patience = 30

        for epoch in range(1, 301):
            # -- train --
            model.train()
            epoch_loss = 0.0
            epoch_samples = 0
            for xb, yb in trn_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
                epoch_samples += len(xb)
            scheduler.step()
            train_loss = epoch_loss / epoch_samples

            # -- validate --
            model.eval()
            with torch.no_grad():
                val_preds = model(val_Xt)
                val_loss = criterion(val_preds, val_yt).item()

            # early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 20 == 0 or epoch == 1:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch:3d} | "
                    f"train_loss={train_loss:.5f} | "
                    f"val_loss={val_loss:.5f} | "
                    f"best_val={best_val_loss:.5f} | "
                    f"lr={lr_now:.2e} | "
                    f"patience={patience_counter}/{patience}"
                )

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        # restore best checkpoint for this run
        model.load_state_dict(best_epoch_state)
        all_state_dicts.append(best_epoch_state)

        # track best single model across all runs
        if best_val_loss < best_single_val_loss:
            best_single_val_loss = best_val_loss
            best_single_state = best_epoch_state

        # test predictions from this run's best model
        model.eval()
        with torch.no_grad():
            run_test_preds = model(test_Xt).cpu().numpy()
        all_test_preds.append(run_test_preds)
        print(f"  Run {run} best val loss: {best_val_loss:.5f}\n")

    # ---- ensemble: average predictions from 5 runs ----
    ensemble_preds = np.mean(all_test_preds, axis=0)
    ensemble_preds_clipped = np.clip(ensemble_preds, 1e-7, 1 - 1e-7)

    # ---- evaluation on test set ----
    test_labels = test_y.astype(int)
    acc = accuracy_score(test_labels, (ensemble_preds >= 0.5).astype(int))
    logloss = log_loss(test_labels, ensemble_preds_clipped)

    # calibration: bin predictions, compare mean pred vs actual freq
    num_bins = 10
    bin_edges = np.linspace(0, 1, num_bins + 1)
    cal_data = []
    for i in range(num_bins):
        mask = (ensemble_preds >= bin_edges[i]) & (ensemble_preds < bin_edges[i + 1])
        if mask.sum() > 0:
            mean_pred = ensemble_preds[mask].mean()
            mean_actual = test_labels[mask].mean()
            cal_data.append((bin_edges[i], bin_edges[i + 1], mask.sum(), mean_pred, mean_actual))

    print(f"\n{'='*50}")
    print("TEST SET EVALUATION (5-run ensemble)")
    print(f"{'='*50}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Log Loss:  {logloss:.5f}")
    print(f"\nCalibration (pred_bin | count | mean_pred | mean_actual):")
    for lo, hi, cnt, mp, ma in cal_data:
        print(f"  [{lo:.1f}, {hi:.1f}) | {cnt:5d} | {mp:.4f} | {ma:.4f}")

    # ---- save best single model ----
    torch.save(best_single_state, model_dir / "mega_nn.pt")
    print(f"\nSaved best single model  -> model/mega_nn.pt")

    # ---- save params + all 5 state dicts ----
    params = {
        "mean": feat_mean,
        "std": feat_std,
        "input_size": input_size,
        "all_states": all_state_dicts,
    }
    with open(model_dir / "mega_nn_params.pkl", "wb") as f:
        pickle.dump(params, f)
    print(f"Saved params & ensemble  -> model/mega_nn_params.pkl")

    results = {
        "accuracy": acc,
        "log_loss": logloss,
        "calibration": cal_data,
        "ensemble_preds": ensemble_preds,
        "num_runs": num_runs,
        "input_size": input_size,
    }
    return results


if __name__ == "__main__":
    results = train()
    print(f"\nDone. Accuracy={results['accuracy']:.4f}, LogLoss={results['log_loss']:.5f}")
