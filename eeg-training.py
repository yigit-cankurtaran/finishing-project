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
