import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from scipy.io import loadmat
from scipy.signal import iirnotch, butter, filtfilt
from tqdm import tqdm # for loop progress
import device_func

# load dataset
mat = loadmat('data/1.mat')

ch_keys = [k for k in mat
           if k.startswith('E') and mat[k].dtype == np.object_]

if 'None' in ch_keys:      # only delete if its actually there
    ch_keys.remove('None')

n_ep  = mat['label'].size
n_ch  = len(ch_keys)
n_samp = mat[ch_keys[0]][0,0].size  # 7500, 15sec*500hz

X = np.empty((n_ep, n_ch, n_samp), dtype=np.float32)
for ci, k in enumerate(ch_keys):
    for ei, cell in enumerate(mat[k][0]):
        X[ei, ci] = cell.squeeze()

y_raw = mat['label'].squeeze()
y = np.where(np.isin(y_raw, [1,11]), 1, 0).astype(np.int64)  # recode

#signal hygiene

fs = 500
def notch(sig, fs=fs, q=30):
    b, a = iirnotch(50, q, fs=fs)      # <- fs=fs
    return filtfilt(b, a, sig, axis=-1)

def band(sig, lo=1, hi=40, fs=fs):
    b, a = butter(4, [lo, hi], btype='band', fs=fs)  # <- fs=fs after btype
    return filtfilt(b, a, sig, axis=-1)

def z(sig):
    m=sig.mean(-1,keepdims=True)
    s=sig.std(-1,keepdims=True)+1e-6
    return (sig-m)/s
X = z(band(notch(X))).astype(np.float32)

# torch, training part

class EEG(Dataset):
    def __init__(self,X,y):
        self.X=torch.tensor(X)
        self.y=torch.tensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.X[i], self.y[i]

ds      = EEG(X,y)
n_train = int(.8*len(ds))
train,val = random_split(ds,[n_train,len(ds)-n_train],
                         generator=torch.Generator().manual_seed(42))
tloader = DataLoader(train,64,shuffle=True)
vloader = DataLoader(val,64)

# 1d cnn
class Net(nn.Module):
    def __init__(self,c_in=35):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(c_in,64,7,padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64,128,5,padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128,256,3,padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256,2)
        )
    def forward(self,x): return self.net(x)

dev = device_func.device_func()
net = Net(X.shape[1]).to(dev)
opt = torch.optim.Adam(net.parameters(),1e-3)
crit= nn.CrossEntropyLoss()

best = 9e9; bad=0; patience=5
for epoch in range(50):
    # train
    net.train()
    for xb,yb in tqdm(tloader, leave=False):
        xb,yb=xb.to(dev),yb.to(dev)
        opt.zero_grad(); loss=crit(net(xb),yb); loss.backward(); opt.step()
    # val
    net.eval(); vloss=hits=tot=0
    with torch.no_grad():
        for xb,yb in vloader:
            xb,yb=xb.to(dev),yb.to(dev)
            out=net(xb)
            vloss+=crit(out,yb).item()*len(yb)
            hits+=(out.argmax(1)==yb).sum().item(); tot+=len(yb)
    vloss/=tot; acc=hits/tot
    print(f'epoch {epoch:02d}  val_loss {vloss:.4f}  acc {acc:.3f}')
    if vloss < best-1e-4:
        best=vloss; bad=0
        torch.save(net.state_dict(),'best_mtouh_cnn.pth')
    else:
        bad+=1
        if bad>=patience:
            print('early stop'); break