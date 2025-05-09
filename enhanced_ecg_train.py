# ecg multilabel classifier on ptb-xl with better preprocessing, sampling, model & training loop

"""
requirements
------------
python -m pip install torch scikit-learn scikit-multilearn wfdb numpy pandas matplotlib tqdm

run
---
python enhanced_ecg_train.py --csv ./ptbxl_database.csv --signals ./signals --scp ./scp_statements.csv
"""

import argparse, os, ast, random, math, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from collections import Counter

import wfdb
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
from skmultilearn.model_selection import iterative_train_test_split
from scipy.signal import iirnotch, butter, filtfilt

import device_func

# utils

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def device_select():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# preprocessing
FS = 500
BP = (0.5, 40)
NOTCH = 50


def spa_raw(sig):
    """band-pass + notch + per-lead z-score, numpy array (t, ch) -> (ch, t) float32"""
    sig = sig.T  # (ch, t)
    b1, a1 = iirnotch(NOTCH, 30, fs=FS)
    b2, a2 = butter(4, BP, btype="band", fs=FS)
    sig = filtfilt(b1, a1, sig, axis=-1)
    sig = filtfilt(b2, a2, sig, axis=-1)
    m = sig.mean(-1, keepdims=True)
    s = sig.std(-1, keepdims=True) + 1e-6
    return ((sig - m) / s).astype(np.float32)


# simple augmentations: noise, shift, drift

def augment(sig):
    if random.random() < 0.5:
        sig += 0.005 * np.random.randn(*sig.shape)
    if random.random() < 0.5:
        shift = random.randint(-100, 100)
        sig = np.roll(sig, shift, axis=-1)
    if random.random() < 0.5:
        drift = np.sin(np.linspace(0, math.pi * 2, sig.shape[-1])) * random.uniform(0.0, 0.1)
        sig += drift
    return sig


# dataset
class ECGDataset(Dataset):
    def __init__(self, X, y, augment_flag=False):
        self.X = X
        self.y = y.astype(np.float32)
        self.augment_flag = augment_flag

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment_flag:
            x = augment(x.copy())
        return torch.from_numpy(x), torch.from_numpy(self.y[idx])


# model
class ResBlock(nn.Module):
    def __init__(self, c, k=7):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, k, padding=k // 2)
        self.bn1 = nn.BatchNorm1d(c)
        self.conv2 = nn.Conv1d(c, c, k, padding=k // 2)
        self.bn2 = nn.BatchNorm1d(c)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class ECGNet(nn.Module):
    def __init__(self, n_ch=12, n_cls=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_ch, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.block1 = ResBlock(64)
        self.block2 = ResBlock(64)
        self.pool = nn.MaxPool1d(4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, n_cls)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


# data loading

def load_scp_map(csv):
    df = pd.read_csv(csv)
    df = df[df["diagnostic"] == 1.0]
    mapping = {row.iloc[0]: row.diagnostic_class for _, row in df.iterrows() if pd.notna(row.diagnostic_class)}
    classes = sorted(set(mapping.values()))
    return mapping, classes


def load_ptbxl(args):
    mapping, classes = load_scp_map(args.scp_csv)
    n_cls = len(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    df = pd.read_csv(args.csv)
    file_col = "filename_hr" if args.sr == 500 else "filename_lr"
    df = df[df[file_col].notna()]

    X, Y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        rec_path = Path(args.sig_dir) / row[file_col]
        try:
            rec = wfdb.rdrecord(rec_path)
            sig = rec.p_signal
        except Exception:
            continue
        try:
            scp_codes = ast.literal_eval(row.scp_codes)
        except Exception:
            continue
        label = np.zeros(n_cls, dtype=np.float32)
        for k in scp_codes:
            if k in mapping:
                label[cls_to_idx[mapping[k]]] = 1.0
        if label.sum() == 0:
            continue
        sig = spa_raw(sig)
        # random crop windows
        if sig.shape[-1] < args.win:
            continue
        start = random.randint(0, sig.shape[-1] - args.win)
        crop = sig[:, start : start + args.win]
        X.append(crop)
        Y.append(label)
    return np.stack(X), np.stack(Y), classes


# training

def trainer(args):
    seed_everything()
    dev = device_select()
    is_mps = dev.type == "mps"

    print("device:", dev)

    if is_mps:
        args.bs = max(2, args.bs // 4)
        use_amp = False
    else:
        use_amp = True

    X, Y, classes = load_ptbxl(args)
    print("data", X.shape, Y.shape)

    # stratified multilabel split
    X_train, y_train, X_val, y_val = iterative_train_test_split(X, Y, test_size=0.2)

    train_ds = ECGDataset(X_train, y_train, augment_flag=True)
    val_ds = ECGDataset(X_val, y_val)

    # imbalance handling
    pos = y_train.sum(0)
    neg = len(y_train) - pos
    pos_weight = torch.tensor(neg / (pos + 1e-6), dtype=torch.float32).to(dev)

    batch = args.bs
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch, num_workers=4)

    net = ECGNet(n_cls=len(classes)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = GradScaler()

    best_f1 = 0.0

    for epoch in range(args.epochs):
        net.train(); running = 0.0
        for xb, yb in tqdm(train_loader, desc=f"train {epoch}"):
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            with autocast(device_type=dev.type, enabled=use_amp):
                out = net(xb)
                loss = criterion(out, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            sched.step()
            running += loss.item()
        print(f"epoch {epoch} train loss {running/len(train_loader):.4f}")

        # validation
        net.eval(); preds = []; gts = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(dev)
                with autocast(device_type="mps"):
                    out = net(xb)
                preds.append(torch.sigmoid(out).cpu())
                gts.append(yb)
        preds = torch.cat(preds).numpy(); gts = torch.cat(gts).numpy()
        bin_preds = (preds > 0.5).astype(np.int8)
        macro_f1 = f1_score(gts, bin_preds, average="macro", zero_division=0)
        print(f"epoch {epoch} macro-f1 {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(net.state_dict(), args.out)
            print("saved best model")


#cli
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--sig_dir", required=True)
    p.add_argument("--scp_csv", required=True)
    p.add_argument("--sr", type=int, default=500)
    p.add_argument("--win", type=int, default=5000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="ecgnet_best.pth")
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    trainer(args)
