import numpy as np
from . import kernel_initializers as kernel

class Layer:
    def __init__(self, units, regularizer=None, name=None):
        self.units = units
        self.name = name
        self.regularizer = regularizer
        
    def forward(self, X):
        return X
    
    def backward(self, grad):
        return grad
    
    def update(self, params):
        return

    def init_params(self, d_in, d_out):
        self.weights = self.kernel_initializer(d_in, d_out)
        self.bias = np.zeros(1)
        self.params = [self.weights, self.bias]
        
    def get_weights(self):
        return self.weights
    
    def set_weights(self, weights):
        self.weights = weights
        
    def get_bias(self):
        return self.bias
    
    def set_bias(self, bias):
        self.bias = bias
    
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    sig = sigmoid(z)
    
    return sig * (1 - sig)

class Dense(Layer):
    def __init__(self, units, regularizer=None, name=None):
        self.units = units
        self.params = []
        self.grads = []
        self.name = name
        self.regularizer = regularizer

class Linear(Dense):        
    def forward(self, X):
        self.X = X
        self.z = np.dot(X, self.weights) + self.bias  # Suma ponderada
        return self.z

    def backward(self, grad_output):
        # Gradientes de los parámetros
        self.dw = np.dot(self.X.T, grad_output)  # dC/dw = dC/dz * dz/dw
        self.db = np.sum(grad_output, axis=0)    # dC/db = dC/dz * dz/db
        self.grads = [self.dw, self.db]
        # Gradiente para las entradas (para la capa anterior)
        grad_input = np.dot(grad_output, self.weights.T)  # dC/dX = dC/dz * dz/dX
        return grad_input
    
    def init_params(self, d_in, d_out):
        self.weights = kernel.he_initialization(d_in, d_out)
        self.bias = np.zeros(1)
        self.params = [self.weights, self.bias]
    
    def update(self, w, b):
        self.weights -= w
        self.bias -= b
    

class ReLU(Dense):      
    def forward(self, X):
        self.X = X
        self.z = np.dot(X, self.weights) + self.bias
        self.a = np.maximum(0, self.z)
        return self.a
    
    def backward(self, grad_output):
        relu_grad = np.where(self.z > 0, 1, 0)
        grad_z = grad_output * relu_grad
        self.dw = np.dot(self.X.T, grad_z)
        self.db = np.sum(grad_z)
        self.grads = [self.dw, self.db]
        grad_input = np.dot(grad_z, self.weights.T)
        return grad_input
    
    def init_params(self, d_in, d_out):
        self.weights = kernel.he_initialization(d_in, d_out)
        self.bias = np.zeros(1)
        self.params = [self.weights, self.bias]
    
    def update(self, w, b):
        self.weights -= w
        self.bias -= b
        
class LeakyReLU(Dense):
    def forward(self, input):
        self.input = input
        self.z = np.dot(input, self.weights) + self.bias
        self.a = np.maximum(0.01 * self.z, self.z)
        return self.a
    
    def backward(self, dL):
        da = np.where(self.input > 0, 1, 0.01)
        dz = dL * da
        self.dw = np.dot(self.input.T, dz)
        self.db = np.sum(dz)
        self.grads = [self.dw, self.db]
        grad_input = np.dot(dz, self.weights.T)
        return grad_input
    
    def init_params(self, d_in, d_out):
        self.weights = kernel.he_initialization(d_in, d_out)
        self.bias = np.zeros(1)
        self.params = [self.weights, self.bias]
    
    def update(self, w, b):
        self.weights -= w
        self.bias -= b
            
class Sigmoid(Dense):       
    def forward(self, X):
        self.X = X
        self.z = np.dot(X, self.weights) + self.bias  # Suma ponderada
        self.a = sigmoid(self.z)  # Aplicar función de activación
        return self.a

    def backward(self, grad_output):
        # Gradiente local de la función Sigmoid
        sigmoid_grad = self.a * (1 - self.a)  # \sigma'(z) -> a es la sigmoide de z, la guardo para evitar calcularla de nuevo

        # Gradiente total respecto a z
        grad_z = grad_output * sigmoid_grad  # dC/dz = dC/da * da/dz
        n, m = self.X.shape
        # Gradientes de los parámetros
        self.dw = np.dot(self.X.T, grad_z) / m # dC/dw = dC/dz * dz/dw
        #self.db = np.sum(grad_z, axis=0)    # dC/db = dC/dz * dz/db
        self.db = np.sum(grad_z) / m
        self.grads = [self.dw, self.db]
        # Gradiente para las entradas (para la capa anterior)
        grad_input = np.dot(grad_z, self.weights.T)  # dC/dX = dC/dz * dz/dX

        return grad_input
    
    def init_params(self, d_in, d_out):
        self.weights = kernel.xavier_initialization(d_in, d_out)
        self.bias = np.zeros(1)
        self.params = [self.weights, self.bias]
    
    def update(self, w, b):
        self.weights -= w
        self.bias -= b

def tanh(z):
    pos_exp = np.exp(z)
    neg_exp = np.exp(-z)
    
    tanh = (pos_exp - neg_exp) / (pos_exp + neg_exp)
    return tanh

class Tanh(Dense):
    def forward(self, X):
        self.X = X
        self.z = np.dot(X, self.weights) + self.bias  # Suma ponderada
        self.a = tanh(self.z)  # Aplicar función de activación
        return self.a

    def backward(self, grad_output):
        # Gradiente local de la función Tanh
        tanh_grad = 1 - (self.a ** 2)  # \sigma'(z) -> a es la sigmoide de z, la guardo para evitar calcularla de nuevo

        # Gradiente total respecto a z
        grad_z = grad_output * tanh_grad  # dC/dz = dC/da * da/dz

        # Gradientes de los parámetros
        self.dw = np.dot(self.X.T, grad_z)  # dC/dw = dC/dz * dz/dw
        #self.db = np.sum(grad_z, axis=0)    # dC/db = dC/dz * dz/db
        self.db = np.sum(grad_z)
        self.grads = [self.dw, self.db]
        # Gradiente para las entradas (para la capa anterior)
        grad_input = np.dot(grad_z, self.weights.T)  # dC/dX = dC/dz * dz/dX

        return grad_input
    
    def init_params(self, d_in, d_out):
        self.weights = kernel.xavier_initialization(d_in, d_out)
        self.bias = np.zeros(1)
        self.params = [self.weights, self.bias]
    
    def update(self, w, b):
        self.weights -= w
        self.bias -= b
        

def softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    logits = np.exp(z)
    return logits / np.sum(logits, axis=1, keepdims=True)

class Softmax(Dense): #as output layer with cross-entropy loss
    def forward(self, X):
        self.X = X
        self.output = softmax(self.X)
    
        return self.output
    
    def backward(self, grad_output):
        self.dw = 0
        self.db = 0

        return grad_output
    
    def init_params(self, d_in, d_out):
        self.weights = np.zeros(shape=(d_in, d_out))
        self.bias = np.zeros(1)
        self.params = [self.weights, self.bias]
        return
    
    def update(self, w, b):
        return 
