import numpy as np

class Neuron:
    def __init__(self):
        pass


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # en vez de n_neurons x n_inputs, para no tener que hacer la transpuesta de W
        self.biases = np.zeros(1, n_neurons)
     
    def forward(self, inputs):
        layer_outputs = []
        #calcular output de toda la capa
        #weights es una matriz con los pesos de toda la capa
        # # -> cada fila de la matriz es un vector con los pesos de cada neurona de la capa
        #biases es un vector con todas las bias de la capa
        self.output = np.dot(inputs, self.weights) + self.biases
        '''
        for neuron_weights, neuron_b  ias in zip(weights, biases): #con zip combina dos listas por elemento ->
        #nueva lista de listas es [[w1, b1], [w2, b2]...[wn, bn]]
            neuron_output = 0
            for n_input, weight in zip(inputs, neuron_weights): #nueva lista de listas es [[i1, w1], [i2, w2]...[in, wn]]
                neuron_output += n_input * weight
            neuron_output += neuron_bias
            layer_outputs.append(neuron_output)'''

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True) #axis=0 suma las columnas, axis=0 suma las filas

def categorical_cross_entropy(y_hat):
    loss = -(np.log(y_hat))
    return loss

class Optimizer: #ajusta los valores de los parámetros para ajustarse bien a los datos -> Gradient Descent
    def __init__(self):
        pass
    
class Loss:
    def calculate(self, output, y):
        return np.mean(self.forward(output, y))
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1: #scalar values
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else: #one-hot encodded
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood