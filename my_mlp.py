import numpy as np

class Sequential:
    def __init__(self, layers):
        #red como lista de capas
        self.layers = layers
        self.stop_training = False

        '''
        for idx, layer in self.layers:
            if idx > 0:
        '''        
        
    def compile(self, loss, optimizer):
        self.loss = loss
        self.loss.net = self
        self.optimizer = optimizer
        self.optimizer.net = self
    
    def fit(self, X, y, epochs, batch_size=32):
        
        loss_history = []
        m, n = X.shape
        
        for epoch in range(epochs):
        #for _ in range(batches):
            batch_idx = np.random.randint(m, size=batch_size)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            y_pred = self.forward(X_batch)
            j = self.loss(y_pred, y_batch)
            self.loss.backward()
            self.optimizer.update(epoch)
            
            '''
            # Check for improvement
            if j < best_loss - min_delta:
                best_loss = j
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch} |", "Loss:", j)
            '''
            loss_history.append(j)
            
        return {
            "loss": loss_history
                }
    
    def forward(self, X):
        #calcular salida del modelo aplicando cada capa secuencialmente
        for layer in self.layers:
            X = layer.forward(X)
        
        return X
        
    def predict(self, X):
        pass
    
    def summary(self):
        for idx, layer in enumerate(self.layers):
            print(f"Layer {idx}: Weights: {layer.weights.shape}, bias: 1, Output shape: (None, {layer.weights.shape[1]}), name: {layer.name}")

    def predict(self, X):
        return self.forward(X)
    
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
        loss = - np.mean(self.target * np.log(self.output + 1e-15) - (1 - self.target) * np.log(1 - self.output + 1e-15))
        return loss
    
    def grad_loss(self): #derivada de funcion de coste
        return self.output - self.target

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
    
#clase optimizer es la que después de calcular los gradientes y la pérdida con la clase loss, actualiza los parámetros a partir de los gradientes
class SGD:
    def __init__(self, net, lr=0.01, epochs=1000, batch_size=32, tolerance=0.0001, lambda_=None):
        self.net = net
        self.lr = lr
        self.epochs = epochs
        self.tolerance = tolerance
        
    def update(self):
        
        for layer in self.net.layers:
            layer.update(
                self.lr * layer.dw,  
                self.lr * layer.db
            )
            
        '''for layer in self.net.layers:
            layer.update([
                params - self.lr * grads
                for params, grads in zip(layer.params, layer.grads)
            ])'''
            
class SGDMomentum:
    def __init__(self, net, lr=0.01, epochs=1000, batch_size=32, tolerance=0.0001, lambda_=None, beta=0.9):
        self.net = net
        self.lr = lr
        self.epochs = epochs
        self.tolerance = tolerance
        self.beta = beta
        
    def update(self):
        for layer in self.net.layers:
            if not hasattr(layer, "VdW"):
                layer.VdW = 0
            if not hasattr(layer, "VdB"):
                layer.VdB = 0
            layer.VdW = self.beta * layer.VdW + (1 - self.beta) * layer.dw
            layer.VdB = self.beta * layer.VdB + (1 - self.beta) * layer.db
            layer.update(
                self.lr * layer.VdW,  
                self.lr * layer.VdB
            )
            
class RMSProp:
    def __init__(self, net, lr=0.01, epochs=1000, batch_size=32, tolerance=0.0001, lambda_=None, beta=0.9):
        self.net = net
        self.lr = lr
        self.epochs = epochs
        self.tolerance = tolerance
        self.beta = beta
        
    def update(self):
        epsilon = 1e-5
        for layer in self.net.layers:
            if not hasattr(layer, "SdW"):
                layer.SdW = 0
            if not hasattr(layer, "SdB"):
                layer.SdB = 0
            layer.SdW = self.beta * layer.SdW + (1 - self.beta) * (layer.dw ** 2)
            layer.SdB = self.beta * layer.SdB + (1 - self.beta) * (layer.db ** 2)
            layer.update(
                self.lr * (layer.dw / np.sqrt(layer.SdW + epsilon)),  #si sdW es muy grande, al dividir por él el resultado es menor -> La actualición del parámetro es más pequeña. Y viceversa
                self.lr * (layer.db / np.sqrt(layer.SdB + epsilon))
            )

class Adam:
    def __init__(self, net, lr=0.01, epochs=1000, batch_size=32, beta1=0.9, beta2=0.999):
        self.net = net
        self.lr = lr
        self.epochs = epochs
        self.beta1 = beta1
        self.beta2 = beta2
        
    def update(self, iter=1):
        epsilon = 1e-5
        for layer in self.net.layers:
            if not hasattr(layer, "VdW"):
                layer.VdW = 0
            if not hasattr(layer, "VdB"):
                layer.VdB = 0
            if not hasattr(layer, "SdW"):
                layer.SdW = 0
            if not hasattr(layer, "SdB"):
                layer.SdB = 0
                
            #Momentum
            layer.VdW = self.beta1 * layer.VdW + (1 - self.beta1) * layer.dw #en primera interación, valor del primer término es 0. 
                                                                                #En primeras iteraciones, el valor general es muy pequeño, tarda en "arrancar"
            layer.VdB = self.beta1 * layer.VdB + (1 - self.beta1) * layer.db
            #RMSProp
            layer.SdW = self.beta2 * layer.SdW + (1 - self.beta2) * (layer.dw ** 2)
            layer.SdB = self.beta2 * layer.SdB + (1 - self.beta2) * (layer.db ** 2)
            
            # correccion para compensar arranque lento
            layer.VdW = layer.VdW / ((1 - self.beta1) ** iter + 1)
            layer.VdB = layer.VdB / ((1 - self.beta1) ** iter + 1)
            layer.SdW = layer.SdW / ((1 - self.beta2) ** iter + 1)
            layer.SdB = layer.SdB / ((1 - self.beta2) ** iter + 1)
            
            layer.update(
                self.lr * (layer.VdW / np.sqrt(layer.SdW + epsilon)), #Momentum / RMSProp
                self.lr * (layer.VdB / np.sqrt(layer.SdB + epsilon))
            )