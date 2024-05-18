import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def parse_delta(masks, seq_len, feature_num):
    """generate deltas from masks, used in BRITS"""
    deltas = []
    for h in range(seq_len):
        if h == 0:
            deltas.append(np.zeros(feature_num))
        else:
            deltas.append(np.ones(feature_num) + (1 - masks[h]) * deltas[-1])
    return np.asarray(deltas)