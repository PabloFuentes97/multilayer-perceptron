import numpy as np

def accuracy(y_pred, y_true):
    return (y_pred == y_true).mean()