import numpy as np

class Network:
    def __init__(self, layers):
        #red como lista de capas
        self.layers = layers
        
    def forward(self, X, y):
        #calcular salida del modelo aplicando cada capa secuencialmente
        for layer in self.layers:
            X = layer.forward(X)
        
        return X
        
    def predict(self, X):
        pass

class Layer:
    def __init__(self, d_in, d_out, activation_func):
        self.weights = np.zeros(shape=(d_in, d_out))
        self.bias = np.zeros(d_in)
        self.activation_func = activation_func #hacerlo un objeto para que contenga tanto función como derivadas
    def forward(self, X):
        #calcular suma
        weighted_sum = np.dot(X.T, self.weights) + self.bias
        #calcular función de activación
        return self.activation_func(weighted_sum)
        
    def backward(self, grad_output):
        #si es capa de salida -> calcula error global -> función de coste
        #si es capa oculta -> gradientes
        #salida * derivada de función de activación * entrada de la neurona
        #cada capa calcula sus gradientes y los devuelve para capas anteriores -> backpropagation
        pass
    def update(self, params):
        #actualizar parámetros con lo que de el optimizer
        pass

class Optimizer:
    def __init__(self, network, lr=0.01, epochs=1000):
        self.network = network
        self.lr = lr
        self.epochs = epochs
    
    def train(self, X, y):
        for _ in range(self.epochs):
            #1) feedforward-> propagación hacia adelante
            #2) Backpropagation-> Calcula error y los gradientes
            #3) Actualiza los pesos
            pass

class ActivationFunc:
    def __init__(self, func, deriv):
        self.func = func
        self.deriv = deriv

def ReLU(X):
    return np.maximum(0, X)

class SGD:
    def __init__(self, net, lr=0.1, epochs=1000, tolerance=0.0001):
        self.net = net
        self.lr = lr
        
    def update(self):
        pass

if __name__ == "__main__":
    net = Network([
        Layer(3, 2, ReLU),
        Layer(3, 3, ReLU),
        Layer(3, 2, Sigmoid)
    ])