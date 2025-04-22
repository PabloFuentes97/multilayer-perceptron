import numpy as np

def he_initialization(d_in, d_out):
    return np.random.randn(d_in, d_out) * np.sqrt(2. / d_in)

def xavier_initialization(d_in, d_out):
    return np.random.randn(d_in, d_out) * np.sqrt(1. / d_in)