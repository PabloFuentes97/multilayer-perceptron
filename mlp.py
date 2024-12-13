import numpy as np

class Layer:
    def __init__(self, units):
        self.units = units
        self.params = [] #lo que tiene que actualizar el optimizador
        self.grads = []
        self.weights = None
        self.bias = None
        
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
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    sig = sigmoid(z)
    
    return sig * (1 - sig)
class Sigmoid(Layer):        
    def forward(self, X):
        
        self.weights = np.zeros(shape=(X.shape[1], self.units))
        self.bias = np.zeros(self.units)
        self.params = [] #lo que tiene que actualizar el optimizador
        self.grads = []
        self.local_grads = []
        
        self.X = X
        self.params = [self.weights, self.bias]
        #en el forward, hay que guardar cualquier cálculo o valor intermedio que sea necesario luego para el backward
        #-> se cachea en la memoria para acceder/consumir a ello during backpropagation
        #self.params = [self.w, self.b]
        z = np.dot(X, self.weights) + self.bias #suma ponderada -> pesos * x; el cell body sum
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
        
        grad = sigmoid_deriv(self.X) * grad_output #derivada parcial de función de activación sigmoid -> gradiente local -> cuánto influyen mis entradas en mi output
        #derivada del sigmoid -> 
                                    #gradientes locales -> cuánto influyen las entradas en el resultado de la función de activación
        self.dw = grad * self.X
        self.db = grad
        
        self.grads = (self.dw, self.db)
        return grad  #gradiente local * gradiente "global" de capas posteriores -> cuánto ha influido mi output en el resultado/coste final

class Softmax:
    def forward(self, X):
        #weighted sum
        z = np.dot(X, self.weights) + self.bias
        #activation function -> softmax
        self.output = np.exp(z) / np.sum(z)
        return self.output
    
    def backward(self, output_gradient):
        n = np.size(self.output)
        z = np.tile(self.output, n)
        grad = np.dot(z * (np.identity - z.T), output_gradient)
        self.dw = grad * self.X
        self.db = grad
        self.grads = (self.dw, self.db)
        return grad

#estructura loss es la que al final del forward, calcula el coste final, y realiza backpropagation
class Loss():
    def __init__(self):
        pass
        
    def backward(self, net):
        #derivada de la loss function con respecto a salida de la red
        grad = self.grad_loss()
        #BACKPROPAGATION
        for layer in reversed(net.layers):
            grad = layer.backward(grad)

class BinaryCrossEntropy(Loss):
    def __call__(self, output, target):
        self.output, self.target = output, target.reshape(output.shape)
        loss = - np.mean(self.target * np.log(self.output) - (1 - self.target) * np.log(1 - self.output))
        return loss.mean()
    
    def grad_loss(self): #derivada de funcion de coste
        return self.output - self.target


def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

#clase optimizer es la que después de calcular los gradientes y la pérdida con la clase loss, actualiza los parámetros a partir de los gradientes
class SGD:
    def __init__(self, lr=0.1, epochs=1000, tolerance=0.0001):
        self.lr = lr
        self.epochs = epochs
        
    def update(self, net):
        for _ in range(self.epochs):
            
            for layer in net.layers:
                layer.weights -= layer.dw * self.lr
                layer.bias -= layer.db * self.lr
            
        '''for layer in self.net.layers:
            layer.update([
                params - self.lr * grads
                for params, grads in zip(layer.params, layer.grads)
            ])'''


class MultilayerPerceptron:
    def __init__(self, layers, loss, optimizer):
        #red como lista de capas
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        
    def forward(self, X):
        #calcular salida del modelo aplicando cada capa secuencialmente
        for layer in self.layers:
            X = layer.forward(X)
        
        return X
        
    def backward(self, loss_output):
        for layer in reversed(self.layers):
            loss_output = layer.backward(loss_output)

    def predict(self, X):
        return self.forward(X)

    def fit(self, X, y):
        for _ in range(self.optimizer.epochs):
            y_pred = self.forward(X)
            self.loss(y_pred, y)
            self.loss.backward()
            self.optimizer.update()
            
    def summary(self):
        for idx, layer in enumerate(self.layers):
            print(f"Layer {idx}: Output shape: ({None if not layer.weights else layer.weights.shape[0]}, {layer.units})")
        
        
if __name__ == "__main__":
    net = MultilayerPerceptron(layers=[
            Sigmoid(24, 24),
            Sigmoid(24, 12),
            Sigmoid(12, 2)
        ],
        loss=BinaryCrossEntropy(),
        optimizer=SGD()                              
    )
    net.summary()