"""
PC03 - 1D-CNN Pipeline (PyTorch)
==================================
Fixes the all-label-3 problem by:
  1. Using a 1D-CNN that learns temporal patterns directly from raw signals
  2. Oversampling minority classes during training
  3. Using focal loss to focus on hard/rare examples

Signals used : acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z  (100 Hz)
Labels       : 10,000 Hz -> downsampled to 100 Hz by majority vote
Window       : 200 samples @ 100 Hz = 2 seconds
Stride       : 50 samples (75% overlap)
Output       : one CSV per test file in Data-Predictions/

Install deps : pip install torch h5py numpy scipy scikit-learn
"""

import os, glob, warnings
import numpy as np
import h5py
from scipy.stats import mode as scipy_mode
from sklearn.utils import shuffle
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TRAIN_H5_DIR  = "./Train"
TEST_H5_DIR   = "./Test"
PRED_DIR      = "./Data-Predictions"
MODEL_PATH    = "./cnn_model.pt"

WINDOW_SIZE   = 200
STRIDE        = 50
N_CHANNELS    = 6
N_CLASSES     = 4

BATCH_SIZE    = 256
EPOCHS        = 30
LR            = 1e-3
RANDOM_STATE  = 42

ACC_RATE      = 100
LABEL_RATE    = 10_000
DOWNSAMPLE    = LABEL_RATE // ACC_RATE

DEVICE = torch.device("mps"  if torch.backends.mps.is_available()  else
                       "cuda" if torch.cuda.is_available()          else
                       "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_h5(path):
    with h5py.File(path, 'r') as f:
        keys    = ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']
        arrays  = {k: f[k][:] for k in keys}
        T       = min(len(v) for v in arrays.values())
        signals = np.stack([arrays[k][:T] for k in keys], axis=1).astype(np.float32)

        if 'label' not in f:
            return signals, None

        labels_hi = f['label'][:]
        n_blocks  = min(T, len(labels_hi) // DOWNSAMPLE)
        labels    = np.zeros(n_blocks, dtype=np.int64)
        for i in range(n_blocks):
            block     = labels_hi[i * DOWNSAMPLE : (i+1) * DOWNSAMPLE]
            labels[i] = int(scipy_mode(block, keepdims=True).mode[0])
        return signals[:n_blocks], labels


def make_windows(signals, labels=None):
    T = signals.shape[0]
    X, y = [], []
    for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        X.append(signals[start:end].T)   # (6, W)
        if labels is not None:
            lbl = int(scipy_mode(labels[start:end], keepdims=True).mode[0])
            y.append(lbl)
    X = np.array(X, dtype=np.float32)
    if labels is not None:
        return X, np.array(y, dtype=np.int64)
    return X


# ─────────────────────────────────────────────
# NORMALISATION
# ─────────────────────────────────────────────

class ChannelNorm:
    def fit(self, X):
        self.mean = X.mean(axis=(0,2), keepdims=True)
        self.std  = X.std(axis=(0,2),  keepdims=True) + 1e-8
        return self
    def transform(self, X):
        return (X - self.mean) / self.std
    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)
    def load(self, path):
        d = np.load(path)
        self.mean, self.std = d['mean'], d['std']
        return self


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class WindowDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y) if y is not None else None
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return (self.X[i], self.y[i]) if self.y is not None else self.X[i]


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class CNN1D(nn.Module):
    def __init__(self, n_channels=6, n_classes=4):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(n_channels, 64,  kernel_size=7, padding=3),
            nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,  128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )
    def forward(self, x):
        return self.head(self.gap(self.blocks(x)))


# ─────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight
    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt   = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train():
    train_files = sorted(glob.glob(os.path.join(TRAIN_H5_DIR, "*.h5")))
    if not train_files:
        raise FileNotFoundError(f"No .h5 files in {TRAIN_H5_DIR}")

    print(f"Found {len(train_files)} training files. Loading ...")
    all_X, all_y = [], []
    for i, path in enumerate(train_files):
        signals, labels = load_h5(path)
        X, y = make_windows(signals, labels)
        all_X.append(X); all_y.append(y)
        print(f"  [{i+1:2d}/{len(train_files)}] {os.path.basename(path)}"
              f"  windows={len(y)}  dist={np.bincount(y, minlength=4)}")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    # Normalise
    norm  = ChannelNorm().fit(X_all)
    X_all = norm.transform(X_all)
    norm.save("./norm_stats.npz")

    # Oversample minority classes to match majority
    counts  = Counter(y_all.tolist())
    max_cnt = max(counts.values())
    xs, ys  = [X_all], [y_all]
    for cls, cnt in counts.items():
        if cnt < max_cnt:
            idx    = np.where(y_all == cls)[0]
            repeat = max_cnt // cnt - 1
            if repeat > 0:
                xs.append(X_all[np.tile(idx, repeat)])
                ys.append(np.full(len(idx) * repeat, cls, dtype=np.int64))
    X_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    X_all, y_all = shuffle(X_all, y_all, random_state=RANDOM_STATE)

    print(f"\nAfter oversampling: {len(y_all)} windows")
    print(f"Label dist: {np.bincount(y_all)}")

    # Class weights for focal loss
    cnt2       = np.bincount(y_all)
    w          = 1.0 / (cnt2 + 1e-6)
    w          = w / w.sum() * N_CLASSES
    cls_weight = torch.tensor(w, dtype=torch.float32).to(DEVICE)

    loader = DataLoader(WindowDataset(X_all, y_all),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model     = CNN1D(N_CHANNELS, N_CLASSES).to(DEVICE)
    criterion = FocalLoss(gamma=2.0, weight=cls_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\nTraining 1D-CNN for {EPOCHS} epochs on {DEVICE} ...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += len(yb)
        scheduler.step()
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={total_loss/total:.4f}  acc={100*correct/total:.1f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    return model, norm


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

def predict_all(model=None, norm=None):
    if model is None:
        model = CNN1D(N_CHANNELS, N_CLASSES).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        norm  = ChannelNorm().load("./norm_stats.npz")
    model.eval()

    test_files = sorted(glob.glob(os.path.join(TEST_H5_DIR, "*.h5")))
    if not test_files:
        raise FileNotFoundError(f"No .h5 files in {TEST_H5_DIR}")

    os.makedirs(PRED_DIR, exist_ok=True)
    print(f"\nGenerating predictions for {len(test_files)} test files ...")

    for path in test_files:
        basename = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(PRED_DIR, basename + ".csv")

        signals, _ = load_h5(path)
        T          = signals.shape[0]
        X_wins     = norm.transform(make_windows(signals))

        pred_labels = []
        with torch.no_grad():
            for xb in DataLoader(WindowDataset(X_wins), batch_size=512, num_workers=0):
                pred_labels.append(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
        pred_labels = np.concatenate(pred_labels)

        # Aggregate overlapping window predictions per sample
        per_sample = np.zeros(T, dtype=np.float64)
        counts     = np.zeros(T, dtype=np.int32)
        for idx, start in enumerate(range(0, T - WINDOW_SIZE + 1, STRIDE)):
            end = start + WINDOW_SIZE
            per_sample[start:end] += pred_labels[idx]
            counts[start:end]     += 1
        counts     = np.maximum(counts, 1)
        per_sample = np.clip(np.round(per_sample / counts), 0, 3).astype(np.int64)

        # Fill tail
        last_start = ((T - WINDOW_SIZE) // STRIDE) * STRIDE + WINDOW_SIZE
        if last_start < T:
            per_sample[last_start:] = per_sample[last_start - 1]

        # Upsample 100 Hz -> 10,000 Hz
        upsampled = np.repeat(per_sample, DOWNSAMPLE)

        # Match exact CSV length
        dummy = os.path.join(PRED_DIR, basename + ".csv")
        target = sum(1 for _ in open(dummy)) if os.path.exists(dummy) else len(upsampled)

        if len(upsampled) >= target:
            final = upsampled[:target]
        else:
            final = np.concatenate([upsampled,
                                    np.full(target - len(upsampled), per_sample[-1])])

        np.savetxt(out_path, final.astype(int), fmt='%d')
        dist = np.bincount(final, minlength=4)
        print(f"  {basename}: {len(final)} rows  dist={dist}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--train-only',   action='store_true')
    p.add_argument('--predict-only', action='store_true')
    args = p.parse_args()

    if args.predict_only:
        predict_all()
    else:
        model, norm = train()
        if not args.train_only:
            predict_all(model, norm)
