import numpy as np

class Loss():
    def __init__(self):
        return
class BinaryCrossEntropy(Loss):
    def __call__(self, y_pred, y_true):
        loss = - np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        return loss
    
    def grad_loss(self, y_pred, y_true): #derivada de funcion de coste
        return y_pred - y_true
    
    def backward(self, grad_loss):
        #derivada de la loss function con respecto a salida de la red
        grad = grad_loss
        #BACKPROPAGATION
        for layer in reversed(self.net.layers):
            grad = layer.backward(grad)

class CategoricalCrossEntropy(Loss):
    def __call__(self, y_pred, y_true):
        self.output, self.target = y_pred, y_true

        return - np.sum(y_true * np.log(y_pred + 1e-15))  # Evitar log(0)
    
    def grad_loss(self):
        return self.output - self.target