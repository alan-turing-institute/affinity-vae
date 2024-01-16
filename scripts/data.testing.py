# testing the data loader function in isolation

import numpy as np
from avae.data import load_data


# ========== testing running the data loader ===============

np.random.seed(42)

datapath = "/Users/ep/Documents/1_datasets/aff_vae/affinity-vae/tutorials/mnist_data/images_train/"


trains, vals, tests, lookup, data_dim = load_data(
    datapath=datapath,
    datatype='npy',
    # lim=lim,
    # splt=splt,
    # batch_s=batch_s,
    # no_val_drop=no_val_drop,
    eval=False,
    affinity=None,
    classes=None
    # gaussian_blur=gaussian_blur,
    # normalise=normalise,
    # shift_min=shift_min,
    # rescale=rescale,
)

print(len(trains))
