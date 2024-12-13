import numpy as np

class MultilayerPerceptron:
    def __init__(self, layers):
        #red como lista de capas
        self.layers = layers
        
    def forward(self, X):
        #calcular salida del modelo aplicando cada capa secuencialmente
        for layer in self.layers:
            X = layer.forward(X)
        
        return X
        
    def predict(self, X):
        pass
    
    def summary(self):
        for idx, layer in enumerate(self.layers):
            print(f"Layer {idx}: Weights: {layer.weights.shape}, bias: 1, Output shape: (None, {layer.weights.shape[1]})")

class Layer:
    def __init__(self, d_in, d_out):
        self.weights = np.zeros(shape=(d_in, d_out))
        #self.bias = np.zeros(d_in)
        self.bias = 0
        self.params = [] #lo que tiene que actualizar el optimizador
        self.grads = []
        
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


class Sigmoid:
    def __init__(self, d_in, d_out):
        # Inicializar pesos, bias, y caches
        self.weights = np.random.randn(d_in, d_out) * 0.01
        self.bias = np.zeros(d_out)
        self.params = [self.weights, self.bias]  # Referencia para optimizadores
        self.grads = []  # Gradientes que actualizará el optimizador

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
        sigmoid_grad = self.a * (1 - self.a)  # \sigma'(z)

        # Gradiente total respecto a z
        grad_z = grad_output * sigmoid_grad  # dC/dz = dC/da * da/dz

        # Gradientes de los parámetros
        self.dw = np.dot(self.X.T, grad_z)  # dC/dw = dC/dz * dz/dw
        self.db = np.sum(grad_z, axis=0)    # dC/db = dC/dz * dz/db
        self.grads = [self.dw, self.db]

        # Gradiente para las entradas (para la capa anterior)
        grad_input = np.dot(grad_z, self.weights.T)  # dC/dX = dC/dz * dz/dX
        return grad_input

'''
class Sigmoid(Layer):        
    def forward(self, X):
        
        self.weights = np.zeros(shape=(d_in, d_out))
        self.bias = np.zeros(d_in)
        self.params = [] #lo que tiene que actualizar el optimizador
        self.grads = []
        self.local_grads = []
        
        print("Forward prop, layer sigmoid")
        
        self.X = X
        self.params = [self.weights, self.bias]
        #en el forward, hay que guardar cualquier cálculo o valor intermedio que sea necesario luego para el backward
        #-> se cachea en la memoria para acceder/consumir a ello during backpropagation
        #self.params = [self.w, self.b]
        z = np.dot(X, self.weights) + self.bias #suma ponderada -> pesos * x; el cell body sum
        self.z = z
        a = sigmoid(z) #aplicar sigmoid sobre suma ponderada -> función de activación -> el firing rate
        
        return a #devuelvo resultado de función de activación
      
    def backward(self, grad_output):
        #calcular cuánto influyen las entradas en el output -> 
        # perceptrón tiene dos pasos ->
        # 1) z = suma ponderada -> mx + b -> derivadas parciales de z respecto a m y b
        # a) dz / dm = x
        # b) dz / db = 1
        # 2) A = función de activación sigmoid(z) -> toma como input z -> derivada de sigmoid respecto a z ->
        # dA / dz = A * (1 - A)
        #regla de la cadena -> 
        #gradiente de los pesos -> dA/dw = dA/dz * dz/dw
        #gradiente de las bias -> dA/db = dA/dz * dz/db
        
        
        grad = sigmoid(self.z) * (1 - sigmoid(self.z)) * grad_output #derivada parcial de función de activación sigmoid -> gradiente local -> cuánto influyen mis entradas en mi output

        #gradientes locales -> cuánto influyen las entradas en el resultado de la función de activación
        self.dw = grad * self.X
        self.db = grad
        
        self.grads = [self.dw, self.db]
        return grad  #gradiente local * gradiente "global" de capas posteriores -> cuánto ha influido mi output en el resultado/coste final
'''

def softmax(z):
    logits = np.exp(z)
    return logits / np.sum(logits)

'''def softmax_derivative(output):
    """
    Calcula la matriz Jacobiana de la función softmax para matrices y vectores.
    
    Si `output` es un vector, devuelve un Jacobiano de nxn.
    Si `output` es una matriz (batch_size x n), devuelve un array (batch_size x n x n).
    """
    if output.ndim == 1:  # Caso de un solo ejemplo (vector)
        n = output.shape[0]
        jacobian = np.diag(output) - np.outer(output, output)
    elif output.ndim == 2:  # Caso de batch (matriz)
        batch_size, n = output.shape
        jacobian = np.zeros((batch_size, n, n))
        for b in range(batch_size):
            out = output[b]
            jacobian[b] = np.diag(out) - np.outer(out, out)
    else:
        raise ValueError("Output debe ser un vector o una matriz.")
    return jacobian'''

class Softmax(Layer):
    def forward(self, X):
        #weighted sum
        self.X = X
        epsilon = 1e-5
        self.params = [self.weights, self.bias]
        self.z = np.dot(X, self.weights) + self.bias
        #activation function -> softmax
        tmp = np.exp(self.z)
        self.output = tmp / np.sum(tmp) + epsilon
        return self.output
    
    
    def backward(self, output_gradient):
        '''
        n = np.size(self.output)
        tmp = np.tile(self.output, n)'''
        
        grad = np.dot(softmax_derivative(self.output), output_gradient)
        self.dw = np.dot(self.X.T, grad)
        self.db = np.sum(grad, axis=0, keepdims=True)
        self.grads = [self.dw, self.db]
        grad_input = np.dot(grad, self.weights.T)  # dC/dX = dC/dz * dz/dX
        return grad_input


#estructura loss es la que al final del forward, calcula el coste final, y realiza backpropagation
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
        loss = - np.mean(self.target * np.log(self.output) - (1 - self.target) * np.log(1 - self.output))
        return loss.mean()
    
    def grad_loss(self): #derivada de funcion de coste
        return self.output - self.target

class CategoricalCrossEntropy(Loss):
    def __call__(self, y_pred, y_true):
        self.output, self.target = y_pred, y_true.reshape(y_pred.shape)
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
    
#clase optimizer es la que después de calcular los gradientes y la pérdida con la clase loss, actualiza los parámetros a partir de los gradientes
def binary_cross_entropy(y_true, y_pred):
    
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def softmax(logits):
    """
    Calcula el softmax de un vector de logits.
    Args:
        logits (np.ndarray): Vector de logits (tamaño N).
    Returns:
        np.ndarray: Vector de probabilidades (tamaño N).
    """
    exp_logits = np.exp(logits - np.max(logits))  # Estabilidad numérica
    return exp_logits / np.sum(exp_logits)

def categorical_cross_entropy(y_true, y_pred):
    """
    Calcula la pérdida categorical cross-entropy.
    Args:
        y_true (np.ndarray): Vector de etiquetas verdaderas (one-hot, tamaño N).
        y_pred (np.ndarray): Vector de predicciones (probabilidades, tamaño N).
    Returns:
        float: Pérdida categorical cross-entropy.
    """
    return -np.sum(y_true * np.log(y_pred + 1e-15))  # Evitar log(0)

def grad_softmax_cross_entropy(y_true, logits):
    """
    Calcula el gradiente de la pérdida categorical cross-entropy respecto a los logits.
    Args:
        y_true (np.ndarray): Vector de etiquetas verdaderas (one-hot, tamaño N).
        logits (np.ndarray): Vector de logits (tamaño N).
    Returns:
        np.ndarray: Gradiente respecto a los logits.
    """
    y_pred = softmax(logits)
    return y_pred - y_true  # Gradiente simplificado

class SGD:
    def __init__(self, net, lr=0.01, epochs=1000, tolerance=0.0001):
        self.net = net
        self.lr = lr
        self.epochs = epochs
        self.tolerance = tolerance
        
    def update(self):
        for layer in self.net.layers:
            layer.weights -= self.lr * layer.dw  
            layer.bias -= self.lr * layer.db
            
        '''for layer in self.net.layers:
            layer.update([
                params - self.lr * grads
                for params, grads in zip(layer.params, layer.grads)
            ])'''

if __name__ == "__main__":
    net = MultilayerPerceptron([
        Sigmoid(24, 24),
        Sigmoid(24, 24),
        Softmax(24, 2)
    ])
    
    optimizer = SGD(net, lr=0.1)
    loss = BinaryCrossEntropy(net)
    
    epochs = 100
    for _ in range(epochs):
        y_pred = net.forward(x)
        loss(y_pred, y)
        loss.backward()
        optimizer.update()