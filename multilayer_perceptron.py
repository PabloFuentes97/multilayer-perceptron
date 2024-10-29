import numpy as np

class Perceptron:
    def __init__(self, X, W):
        self.X = X #entradas
        self.W = W #pesos de entradas
    
    def phi(self, Z):
        pass
        
    def fit(self):
        Z = np.dot(self.W, self.X)
        y = self.phi(Z)
        
        return y

class Layer():
    def __init__(self):
        self.params = []
        self.grads = []
    
    def __call__(self, x):
        #por defecto devuelve los inputs, cada capa de clase derivada hace una tarea distinta
        return x
    def backward(self, grad):
        #cada capa calcula sus gradientes en base a los gradientes de las capas anteriores y los devuelve para las capas siguientes
        return grad
    def update(self, params):
        #si hay parámetros, se actualizan con lo que devuelva el optimizer
        return

#clases derivadas de Layer
class Linear(Layer):
    def __init__(self, d_in, d_out):
        #pesos de la capa
        #d_in -> número de conexiones de entrada
        #d_out -> número de conexiones de salida
        self.w = np.zeros(shape=(d_in, d_out))
        
        self.b = np.zeros(d_out)
    #esto es el forward
    def __call__(self, X):
        self.X = X
        self.params = [self.w, self.b]
        #salida del perceptrón
        return np.dot(X, self.w) + self.b
    
    def backward(self, grad_output):
        #gradientes para capa siguiente (BACKPROPAGATION)
        grad = np.dot(grad_output, self.w.T) #derivada parcial
        self.grad_w = np.dot(self.X.T, grad_output)
        self.grad_b = grad_output.mean(axis=0) * self.X.shape[0]
        self.grads = [self.grad_w, self.grad_b]
        
        return grad

class Sigmoid(Layer):
    def __call__(self, x):
        self.x = x
        return sigmoid(x)
    
    def backward(self, grad_output):
        grad = sigmoid(self.x) * (1 - sigmoid(self.x))
        return grad_output * grad

class MLP:
    def __init__(self, layers):
        #MLP es solo una lista de capas
        self.layers = layers
        
    def __call__(self, X):
        #se calcula la salida del modelo aplicando cada capa secuencialmente
        for layer in self.layers:
            X = layer(X)
        
        return X

class MultiLayerPerceptron:
    #ejemplo con una sola capa oculta
    def __init__(self, D_in, H, D_out, loss, grad_loss, activation):
        self.w1, self.b1 = 0 #peso y bias de capa oculta
        self.w2, self.b2 = 0 #peso y bias de capa de salida
        
        self.ws = []
        self.loss = loss #funcion de coste
        self.grad_loss = grad_loss #derivada de funcion de coste
        self.activation = activation
    #calcular salida del perceptron
    def __call__(self, X):
        #capa oculta
        self.h_pre = np.dot(self.w1, X) + self.b1 #sumatorio ponderado -> pesos por features
        self.h = relu(self.h_pre) #valor tras función de activación
        
        y_hat = np.dot(self.w2 * self.h) + self.b2 #capa de salida
        return self.activation(y_hat)
    
    def fit(self, X, y, epochs=100, lr=0.001, batch_size=None, verbose=True, log_each=1):
        batch_size = len(X) if batch_size == None else batch_size
        batches = len(X) // batch_size
        l = []
        for e in range(1, epochs + 1):
            _l = []
            for b in range(batches):
                x = X[b * batch_size: (b+1) * batch_size]
                y = y[b * batch_size: (b+1) * batch_size]
                y_pred = self(X) #calcular salida de perceptron, sería como predict o hipótesis
                loss = self.loss(y, y_pred)
                _l.append(loss)
                
                #Backpropagation
                dldy = self.grad_loss(y, y)
                grad_w2 = np.dot(self.h.T, dldy)
                grad_b2 = dldy.mean(axis=0)
                dldh = np.dot(dldy, self.w2.T) * reluPrime(self.h_pre)
                grad_w1 = np.dot(x.T, dldh)
                grad_b1 = dldh.mean(axis=0)
                
                #Actualizar pesos
                self.w1 -= lr * grad_w1
                self.b1 -= lr * grad_b1
                self.w2 -= lr * grad_w2
                self.b2 -= lr * grad_b2
            
            l.append(np.mean(_l))
            self.ws.append((
                self.w1.copy(),
                self.b1.copy(),
                self.w2.copy(),
                self.b2.copy()
            ))
            if verbose and not e % log_each:
                print(f"Epoch: {e}/{epochs}, Loss: {np.mean(l):.5f}")
                
#OPTIMIZER
class SGD():
    def __init__(self, net, lr):
        self.net = net #network
        self.lr = lr
    
    def update(self):
        #ejecutar esta función fuera en un bucle for -> for epoch in range(1000): update()
        for layer in self.net.layers:
            layer.update([
                params - self.lr * grads
                for params, grads in zip(layer.params, layer.grads)
            ])
            
#LOSS FUNCTIONS
class Loss():
    def __init__(self, net):
        self.net = net
        
    def backward(self):
        #calcular derivada de la loss function con respecto a la salida del MLP
        grad = self.grad_loss()
        #BACKPROPAGATION
        for layer in reversed(self.net.layers):
            grad = layer.backward(grad)
            
#clases derivadas de Loss       
class MSE(Loss):
    def __call__(self, output, target):
        self.output, self.target = output, target.reshape(output.shape)
        loss = np.mean((self.output - self.target) ** 2)
        return loss.mean()
    def grad_loss(self):
        return self.output - self.target
    

'''  
network = model.createNetwork([
    layers.DenseLayer(input_shape, activation='sigmoid'),
    layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
    layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
    layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
    layers.DenseLayer(output_shape, activation='softmax', weights_initializer='heUniform')
    ])
model.fit(network, data_train, data_valid, loss='binaryCrossentropy', learning_rate=0.0314, batch_size=8,
epochs=84)
''' 