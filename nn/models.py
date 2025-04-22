import numpy as np
from .train_test_split import train_test_split
from termcolor import colored
import time
import json

class Sequential:
    def __init__(self, input_dim, layers):
        #red como lista de capas
        self.layers = layers
        self.initialize_params(input_dim)
        '''
        for idx, layer in self.layers:
            if idx > 0:
        '''        
    
    def initialize_params(self, input_dim):
        prev_dim = input_dim
        for layer in self.layers:
            layer.init_params(prev_dim, layer.units)
            #layer.init_params(layer.units, prev_dim) / usar asi para np.dot(W.T, X) -> 
            #se almacena W en vectores columnas, cada neurona una columna; al hacer T se ajustan las dimensiones
            prev_dim = layer.units  # La salida de esta capa es la entrada de la siguiente
       
        #version andrew ng -> le cambia el orden -> (unidades de capa actual l, unidades de capa anterior l - 1)
        '''
        for l in range(len(self.layers)):
            layer_weights = np.random.randn(self.layers[l].units, self.layers[l - 1].units)
            layer_bias = np.zeros(shape=(self.layers[l].units))
        '''     
    
    def compile(self, loss, optimizer):
        self.loss = loss
        self.loss.net = self
        self.optimizer = optimizer
        self.optimizer.net = self
    
    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.get_weights())
            
        return weights
    
    def set_weights(self, weights):
        for layer, w  in zip(self.layers, weights):
            w = np.array(w)
            layer.set_weights(w)
    
    def get_bias(self):
        bias = []
        for layer in self.layers:
            bias.append(layer.get_bias())
            
        return bias
    
    def set_bias(self, bias):
        for layer, b in zip(self.layers, bias):
            layer.set_bias(b)
    
    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters.append(layer.params)
            
        return parameters
    
    def forward(self, X):
        #calcular salida del modelo aplicando cada capa secuencialmente
        for layer in self.layers:
            X = layer.forward(X)
    
        return X
    
    def backward(self, grad_loss):
        for layer in reversed(self.layers):
            grad_loss = layer.backward(grad_loss)
        
    def predict(self, X):
        return self.forward(X)
    
    def save(self, filename, format="json"):
        formats_allowed = set()
        formats_allowed.add("json")
        if format not in formats_allowed:
            raise Exception
        
        info = {"weights": [w.tolist() for w in self.get_weights()],
                "bias": [b.tolist() for b in self.get_bias()]}
        with open(filename, 'w') as fp:
            json.dump(info, fp)
        
    def load(self, filename):
        with open(filename, 'r') as fp:
            info = json.load(fp)
        self.set_weights(info["weights"])
        self.set_bias(info["bias"])