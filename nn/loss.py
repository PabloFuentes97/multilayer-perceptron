import numpy as np

class Loss():
    def __init__(self):
        return
class BinaryCrossEntropy(Loss):
    def __call__(self, y_pred, y_true):
        return - np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
    
    def grad_loss(self, y_pred, y_true): #derivada de funcion de coste
        return y_pred - y_true

class CategoricalCrossEntropy(Loss):
    def __call__(self, y_pred, y_true):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))

    
    def grad_loss(self, y_pred, y_true):
        return y_pred - y_true