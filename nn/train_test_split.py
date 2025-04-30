import numpy as np
import pandas as pd
import typing
import copy

def train_test_split(X: np.ndarray, y: np.ndarray, train_size=0.8):
    if train_size >= 1:
        raise Exception
    
    size = len(X)
    lim = int(size * train_size)
    rand_idx = list(np.random.permutation(size))
    train_idx = rand_idx[: lim]
    test_idx = rand_idx[lim:]
    
    train_X = X[train_idx]
    test_X = X[test_idx]
    train_y = y[train_idx]
    test_y = y[test_idx]
    
    return train_X, test_X, train_y, test_y
