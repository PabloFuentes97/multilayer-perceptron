import numpy as np

def train_test_split(X, y, train_size=0.8):
    size = len(X)
    sample_interval = batch_sample = np.random.choice(size, train_size, replace=False)