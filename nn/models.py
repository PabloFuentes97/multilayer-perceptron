import numpy as np
from .train_test_split import train_test_split
from .create_minibatches import create_minibatches
from .layers import *
from .history import History
from .metrics import accuracy
from pydoc import locate
import time
import json

def save(net, filename):   
    info = {"layers": []}
    info["input_dim"] = net.input_dim
    
    for layer in net.layers:
        info_layer = {
            "weights": [w.tolist() for w in layer.get_weights()],
            "bias": [b.tolist() for b in layer.get_bias()],
            "activation": type(layer).__name__,
            "name": layer.name
        }
        info["layers"].append(info_layer)
        
    with open(filename, 'w') as fp:
        json.dump(info, fp)
        
def load(filename):
    with open(filename, 'r') as fp:
        info = json.load(fp)

    net = Sequential(init=False)
    net.input_dim = info["input_dim"]
    for layer_info in info["layers"]:
        layer_type = globals()[layer_info['activation']]
        layer = layer_type()
        layer.set_weights(np.array(layer_info["weights"]))
        layer.set_bias(np.array(layer_info["bias"]))
        net.add_layer(layer)
        
    return net

class Sequential:
    def __init__(self, input_dim=0, layers=[], init=True):
        self.input_dim = input_dim
        self.layers = layers
        if init:
            self.initialize_params(input_dim)  
    
    def add_layer(self, layer, idx=None):
        if idx:
            self.layers.insert(layer, idx)
        else:
            self.layers.append(layer)
    
    def get_layer_byname(self, name):
        for layer in self.layers:
            if layer.name and layer.name == name:
                return layer
        
    def get_layer_byidx(self, idx):
        if idx >= len(self.layers):
            return False
        
        return self.layers[idx]
    
    def initialize_params(self, input_dim):
        prev_dim = input_dim
        for layer in self.layers:
            layer.init_params(prev_dim, layer.units)
            prev_dim = layer.units
       
    def compile(self, criterion, optimizer, metrics=None):
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
    
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
        for layer in self.layers:
            X = layer.forward(X)
    
        return X
    
    def backward(self, grad_loss):
        for layer in reversed(self.layers):
            grad_loss = layer.backward(grad_loss)
        
    def predict(self, X):
        return self.forward(X)
    
    def train_batch_step(self, X, y):
        self.history.on_batch_begin()
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)
        grad_loss = self.criterion.grad_loss(y_pred, y)
        self.backward(grad_loss)
        self.optimizer.update()
        self.history.on_batch_end(y_pred, y, loss, "train")
        
    def train_epoch_step(self, X, y, batch_size):
        mini_batches = create_minibatches(X, y, batch_size)
        mini_batches_num = len(mini_batches)
        for batch_x, batch_y in mini_batches:    
            self.train_batch_step(batch_x, batch_y)
        
        return mini_batches_num
    
    def validation_batch_step(self, X, y):
        self.history.on_batch_begin()
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)
        self.history.on_batch_end(y_pred, y, loss, "validation")
        
    def validation_epoch_step(self, X, y, batch_size):
        mini_batches = create_minibatches(X, y, batch_size)
        mini_batches_num = len(mini_batches)
        for batch_x, batch_y in mini_batches:    
            self.validation_batch_step(batch_x, batch_y)
        
        return mini_batches_num

    def train_epoch(self, X, y, batch_size):
        self.history.on_train_epoch_begin()
        minibatches_num = self.train_epoch_step(X, y, batch_size)
        self.history.on_train_epoch_end(minibatches_num)
        
    def validation_epoch(self, X, y, batch_size):
        self.history.on_validation_epoch_begin()
        minibatches_num = self.validation_epoch_step(X, y, batch_size)
        self.history.on_validation_epoch_end(minibatches_num)
            
    def fit(self, X, y, epochs=100, batch_size=32, validation=None, validation_data=None, validation_batch_size=0, verbose=0, early_stopping=None):
        self.history = History(self.metrics)
        self.history.on_train_begin()
        
        if validation:
            X_cv, y_cv = validation_data
        
        for epoch in range(epochs):
            #training
            before_train_time = time.time()
            self.train_epoch(X, y, batch_size)
            after_train_time = time.time()   

            #validation
            if validation:
                self.validation_epoch(X_cv, y_cv, batch_size=validation_batch_size)
            
            #verbose
            if verbose > 0 and epoch % verbose == 0:
                print(f"Epoch {epoch} | {after_train_time - before_train_time:2f}s | train loss: {self.history.train['loss'][-1]} | validation loss: {self.history.validation['loss'][-1]}")   
            
            #early_stopping
            if early_stopping and early_stopping(self.history.train["loss"][-1]):
                break
        
        self.history.on_train_end()
        return self.history