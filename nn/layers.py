import numpy as np

class Layer:
    def __init__(self, d_in, d_out, name=None):
        self.weights = np.random.randn(d_in, d_out) * 0.01
        #self.bias = np.zeros(d_in)
        self.bias = 0
        self.params = [] #lo que tiene que actualizar el optimizador
        self.grads = []
        self.name = name
        
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

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    sig = sigmoid(z)
    
    return sig * (1 - sig)


class Linear(Layer):        
    def forward(self, X):
        """
        Realiza la propagación hacia adelante.
        X: entrada de la capa (batch_size, d_in)
        """
        self.X = X
        self.z = np.dot(X, self.weights) + self.bias  # Suma ponderada
        return self.z

    def backward(self, grad_output):
        """
        Realiza la propagación hacia atrás.
        grad_output: gradiente del coste respecto a la salida de esta capa
        """

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
    
class Sigmoid(Layer):
    '''
    def __init__(self, d_in, d_out):
        # Inicializar pesos, bias, y caches
        self.weights = np.random.randn(d_in, d_out) * 0.01
        self.bias = np.zeros(d_out)
        self.params = [self.weights, self.bias]  # Referencia para optimizadores
        self.grads = []  # Gradientes que actualizará el optimizador
    '''
    def forward(self, X):
        """
        Realiza la propagación hacia adelante.
        X: entrada de la capa (batch_size, d_in)
        """
        self.X = X
        self.z = np.dot(X, self.weights) + self.bias  # Suma ponderada
        self.a = sigmoid(self.z)  # Aplicar función de activación
        return self.a

    def backward(self, grad_output):
        """
        Realiza la propagación hacia atrás.
        grad_output: gradiente del coste respecto a la salida de esta capa
        """
        # Gradiente local de la función Sigmoid
        sigmoid_grad = self.a * (1 - self.a)  # \sigma'(z) -> a es la sigmoide de z, la guardo para evitar calcularla de nuevo

        # Gradiente total respecto a z
        grad_z = grad_output * sigmoid_grad  # dC/dz = dC/da * da/dz

        # Gradientes de los parámetros
        self.dw = np.dot(self.X.T, grad_z)  # dC/dw = dC/dz * dz/dw
        self.db = np.sum(grad_z, axis=0)    # dC/db = dC/dz * dz/db
        self.grads = [self.dw, self.db]

        # Gradiente para las entradas (para la capa anterior)
        grad_input = np.dot(grad_z, self.weights.T)  # dC/dX = dC/dz * dz/dX

        return grad_input
    
    def update(self, w, b):
        self.weights -= w
        self.bias -= b


def softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    logits = np.exp(z)
    return logits / np.sum(logits)

class Softmax(Layer): #as output layer with cross-entropy loss
    def forward(self, X):
        #weighted sum
        self.X = X
        epsilon = 1e-5
        self.params = [self.weights, self.bias]
        #self.z = np.dot(X, self.weights) + self.bias
        #activation function -> softmax
        exp_logits = np.exp(self.X - np.max(self.X, axis=-1, keepdims=True))  # Estabilidad numérica
        self.output = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
        return self.output
    
    
    def backward(self, output_gradient):
        '''
        n = np.size(self.output)
        tmp = np.tile(self.output, n)'''
        #grad = y_pred - output_gradient
        '''
        
        grad = output_gradient
        self.dw = np.dot(self.X.T, grad)
        self.db = np.sum(grad, axis=0, keepdims=True)
        self.grads = (self.dw, self.db)
        grad_input = np.dot(grad, self.weights.T)  # dC/dX = dC/dz * dz/dX
        return grad_input'''
        self.dw = 0
        self.db = 0
        '''
        batch_size = output_gradient.shape[0]
        grad = (self.output - output_gradient) / batch_size
        grad_input = np.dot(grad, self.weights.T)
        return grad_input'''
        return output_gradient
    
    def update(self, w, b):
        return 
