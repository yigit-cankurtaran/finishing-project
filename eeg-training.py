import scipy.io as sio
mat = sio.loadmat('data/1.mat')      # dict of numpy arrays
print(mat.keys())                    # checking what’s inside

#no idea about the database now:
# possible cases:
#   'eeg'  -> shape (n_seg, n_chan, n_time)
#   'label' or 'y' -> shape (n_seg,)  (0=normal,1=abnormal)
# if you see nested structs, drill down:  mat['eeg'][0,0]['data'] etc.

print(mat['EC3A1'].shape) # prints(1,267)
# 2d shape

chan = mat['EC3A1']          # shape (1,267), dtype=object
epoch0 = chan[0,0]           # grab first cell
print(epoch0.shape)
#prints (1,7500), likely 30s @ 250Hz
