import numpy as np

class Loss():
    def __init__(self, net):
        self.net = net
        
    def backward(self):
        #derivada de la loss function con respecto a salida de la red
        grad = self.grad_loss()
        #BACKPROPAGATION
        for layer in reversed(self.net.layers):
            grad = layer.backward(grad)
    

class BinaryCrossEntropy(Loss):
    def __call__(self, output, target):
        self.output, self.target = output, target.reshape(output.shape)
        loss = - np.mean(self.target * np.log(self.output + 1e-15) - (1 - self.target) * np.log(1 - self.output + 1e-15))
        return loss
    
    def grad_loss(self): #derivada de funcion de coste
        return self.output - self.target
    
    def backward(self):
        #derivada de la loss function con respecto a salida de la red
        grad = self.grad_loss()
        #BACKPROPAGATION
        for layer in reversed(self.net.layers):
            grad = layer.backward(grad)

class CategoricalCrossEntropy(Loss):
    def __call__(self, y_pred, y_true):
        self.output, self.target = y_pred, y_true#.reshape(y_pred.shape)
        """
        Calcula la pérdida categorical cross-entropy.
        Args:
            y_true (np.ndarray): Vector de etiquetas verdaderas (one-hot, tamaño N).
            y_pred (np.ndarray): Vector de predicciones (probabilidades, tamaño N).
        Returns:
            float: Pérdida categorical cross-entropy.
        """
        return -np.sum(y_true * np.log(y_pred + 1e-15))  # Evitar log(0)
    
    def grad_loss(self):
        return self.output - self.target