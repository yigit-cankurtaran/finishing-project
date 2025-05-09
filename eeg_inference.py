"""
usage:
    python eeg_inference.py <epoch.npy|epoch.mat> [ckpt.pth]

prints 'normal' or 'abnormal'
"""

import sys, numpy as np, torch, torch.nn as nn
from scipy.io import loadmat
from scipy.signal import iirnotch, butter, filtfilt
import device_func

FS       = 500
WIN_SAMP = 7500

# hygiene----------
def notch50(sig, fs=FS, q=30):
    b, a = iirnotch(50, q, fs=fs)
    return filtfilt(b, a, sig, axis=-1)

def band(sig, lo=1, hi=40, fs=FS):
    b, a = butter(4, [lo, hi], btype="band", fs=fs)
    return filtfilt(b, a, sig, axis=-1)

def z(sig):
    m = sig.mean(-1, keepdims=True)
    s = sig.std(-1, keepdims=True) + 1e-6
    return (sig - m) / s

spa = lambda x: z(band(notch50(x))).astype(np.float32)

# model
class Net(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, 2)
        )
    def forward(self, x): return self.net(x)

# helper
def load_epoch(path: str, c_expected: int) -> np.ndarray:
    if path.endswith(".npy"):
        arr = np.load(path)
    elif path.endswith(".mat"):
        mat  = loadmat(path)
        keys = sorted([k for k in mat if k.startswith("E") and mat[k].dtype == object])
        arr  = np.stack([mat[k][0, 0].squeeze() for k in keys])
    else:
        raise ValueError("need .npy or .mat")
    assert arr.shape == (c_expected, WIN_SAMP), f"got {arr.shape}, expected {(c_expected, WIN_SAMP)}"
    return arr

# main
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python eeg_inference.py <epoch.npy|epoch.mat> [ckpt.pth]")
        sys.exit(1)
    epoch_path = sys.argv[1]
    ckpt_path  = sys.argv[2] if len(sys.argv) > 2 else "best_mtouh_cnn.pth"

    # sniff channel count from checkpoint
    state  = torch.load(ckpt_path, map_location=device_func.device_func())
    c_in   = next(iter(state.values())).shape[1]

    net = Net(c_in)
    net.load_state_dict(state)
    net.eval()

    raw = load_epoch(epoch_path, c_in)
    x = torch.tensor([spa(raw)]) # (1,c_in,7500)
    with torch.no_grad():
        logits = net(x)
        probs  = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        label = "abnormal" if pred.item() else "normal"
        print(f"{label} - {conf.item()*100:.1f}% confident")
