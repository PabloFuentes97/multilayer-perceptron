import numpy as np

class Layer:
    def __init__(self, units, regularizer=None, name=None):
        '''
        limit = np.sqrt(2 / d_in + d_out)
        self.weights = np.random.uniform(low=-limit, high=limit, size=(d_in, d_out)) #xavier initialization
        #self.weights = np.random.randn(d_in, d_out) * 0.01
        #self.bias = np.zeros(d_in)
        self.bias = 0
        '''
        self.units = units
        self.params = [] #lo que tiene que actualizar el optimizador
        self.grads = []
        self.name = name
        self.regularizer = regularizer
        
    def forward(self, X):
        #calcular suma
        #weighted_sum = np.dot(X.T, self.weights) + self.bias
        #calcular función de activación
        #return self.activation_func(weighted_sum)
        #cada capa hace algo distinto
        
        #calcular gradiente local del nodo ->
        #derivada parcial de función de activación respecto a las entradas -> GRADIENTES LOCALES
        #guardarselo para luego encadenarlo al gradiente "global" que llega de nodos posteriores 
        # -> BACKPROPAGATION
        return X
    
    def backward(self, grad):
        #cada capa calcula sus gradientes y los devuelve para capas anteriores -> backpropagation
        #backpropagation
        return grad
    
    def update(self, params):
        #actualizar parámetros con lo que de el optimizer
        return

    def init_params(self, d_in, d_out):
        #cambiarlo por usar el kernel initializer adecuado
        limit = np.sqrt(2 / d_in + d_out)
        self.weights = np.random.uniform(low=-limit, high=limit, size=(d_in, d_out)) #xavier initialization
        self.bias = 0
        
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
    def __init__(self, d_in, d_out, regularizer=None, name=None):
        limit = np.sqrt(2 / d_in + d_out)
        self.weights = np.random.uniform(low=-limit, high=limit, size=(d_in, d_out)) #xavier initialization
        #self.weights = np.random.randn(d_in, d_out) * 0.01
        #self.bias = np.zeros(d_in)
        self.units = 0
        self.bias = 0
        self.params = [] #lo que tiene que actualizar el optimizador
        self.grads = []
        self.name = name
        
        self.regularizer = regularizer


class Linear(Layer):        
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
    
    def update(self, w, b):
        self.weights -= w
        self.bias -= b

class ReLU(Layer):
    def forward(self, X):
        self.X = X
        self.z = np.dot(X, self.weights) + self.bias
        self.a = np.maximum(0, self.z)
        return self.a
    
    def backward(self, grad_output):
        relu_grad = np.where(self.z > 0, 1, 0)
        grad_z = grad_output * relu_grad #element-wise product
        self.dw = np.dot(self.X.T, grad_z)
        self.db = np.sum(grad_z)
        self.grads = [self.dw, self.db]
        grad_input = np.dot(grad_z, self.weights.T)
        return grad_input
    
    def update(self, w, b):
        reg = 0
        '''if self.regularizer:
            reg = self.regularizer(self.weights)'''
        self.weights -= w + reg
        self.bias -= b
        
class LeakyReLU(Layer):
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
    
    def update(self, w, b):
        reg = 0
        '''if self.regularizer:
            reg = self.regularizer(self.weights)'''
        self.weights -= w + reg
        self.bias -= b
            
class Sigmoid(Layer):
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
    
    def update(self, w, b):
        #meter aqui regularizacion
        reg = 0
        '''if self.regularizer:
            reg = self.regularizer(self.weights)'''
        self.weights -= w + reg
        self.bias -= b

def tanh(z):
    pos_exp = np.exp(z)
    neg_exp = np.exp(-z)
    
    tanh = (pos_exp - neg_exp) / (pos_exp + neg_exp)
    return tanh

class Tanh(Layer):
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
    
    def update(self, w, b):
        #meter aqui regularizacion
        reg = 0
        '''if self.regularizer:
            reg = self.regularizer(self.weights)'''
        self.weights -= w + reg
        self.bias -= b
        

def softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    logits = np.exp(z)
    return logits / np.sum(logits, axis=1, keepdims=True)

class Softmax(Layer): #as output layer with cross-entropy loss
    def forward(self, X):
        self.X = X
        self.params = [self.weights, self.bias]
        #self.z = np.dot(X, self.weights) + self.bias
        #activation function -> softmax
        '''
        exp_logits = np.exp(self.X - np.max(self.X, axis=-1, keepdims=True))  # Estabilidad numérica
        self.output = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)'''
        self.output = softmax(self.X)
    
        return self.output
    
    
    def backward(self, output_gradient):
        self.dw = 0
        self.db = 0
        return output_gradient
    
    def update(self, w, b):
        return 
