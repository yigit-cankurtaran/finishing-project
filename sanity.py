"""
sanity.py
compare cnn prediction vs ground-truth label for one epoch

usage:
    python sanity.py [epoch_idx]

expects:
    1.mat                - original mtouh file
    best_mtouh_cnn.pth   - weights you trained
"""

import sys, numpy as np, torch, torch.nn as nn, scipy.io as sio
from scipy.signal import iirnotch, butter, filtfilt
import device_func

# constants
MAT_PATH  = "data/1.mat"
CKPT_PATH = "best_mtouh_cnn.pth"
FS, WIN   = 500, 7500
EPOCH_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# hygiene funcs
def notch(x): b,a=iirnotch(50,30,fs=FS);                return filtfilt(b,a,x,axis=-1)
def band(x):  b,a=butter(4,[1,40],btype='band',fs=FS);  return filtfilt(b,a,x,axis=-1)
def z(x):     m=x.mean(-1,keepdims=True); s=x.std(-1,keepdims=True)+1e-6; return (x-m)/s
spa = lambda x: z(band(notch(x))).astype(np.float32)

# load epoch
mat   = sio.loadmat(MAT_PATH)
keys  = sorted([k for k in mat if k.startswith('E') and mat[k].dtype == object])
assert len(keys) == 36, f"need 36 chans, got {len(keys)}"

n_ep  = mat['label'].size
if not (0 <= EPOCH_IDX < n_ep):
    raise IndexError(f"epoch {EPOCH_IDX} out of range 0‑{n_ep-1}")

epoch = np.stack([mat[k][0, EPOCH_IDX].squeeze() for k in keys])  # (36,7500)
gt    = 0 if int(mat['label'].squeeze()[EPOCH_IDX]) == 0 else 1    # 0 normal else abnormal

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
    def forward(self, x):
        return self.net(x)

state = torch.load(CKPT_PATH, map_location=device_func.device_func())
c_in  = next(iter(state.values())).shape[1]

net = Net(c_in)
net.load_state_dict(state)
net.eval()

x = torch.from_numpy(spa(epoch)).unsqueeze(0)        # (1,36,7500)
with torch.no_grad():
    probs = torch.softmax(net(x), dim=1)
    conf, pred = torch.max(probs, dim=1)

label  = "abnormal" if pred.item() else "normal"
truth  = "abnormal" if gt else "normal"
print(f"epoch {EPOCH_IDX}: model {label} ({conf.item()*100:.1f} %) | ground-truth {truth}")
