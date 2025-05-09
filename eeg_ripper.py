"""usage:
    python eeg_ripper.py [epoch_idx] [mat_path] [out_prefix]

defaults:
    epoch_idx  = 0
    mat_path   = 1.mat
    out_prefix = epoch
writes <out_prefix><idx>.npy (shape (36,7500))
"""

import sys
import numpy as np
import scipy.io as sio
import pathlib as pl

idx       = int(sys.argv[1]) if len(sys.argv) > 1 else 0
mat_path  = sys.argv[2] if len(sys.argv) > 2 else "data/1.mat"
prefix    = sys.argv[3] if len(sys.argv) > 3 else "epoch"

mat   = sio.loadmat(mat_path)
keys  = sorted([k for k in mat if k.startswith("E") and mat[k].dtype == object])
assert len(keys) == 36, f"expected 36 channels, got {len(keys)}"

n_ep  = mat["label"].size
if not (0 <= idx < n_ep):
    raise IndexError(f"epoch {idx} out of range 0-{n_ep-1}")

epoch = np.stack([mat[k][0, idx].squeeze() for k in keys])  # (36,7500)
out   = f"{prefix}{idx}.npy"
np.save(out, epoch.astype(np.float32))
print(f"saved {out}  shape {epoch.shape}")
