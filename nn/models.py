import numpy as np
from .train_test_split import train_test_split
from .create_minibatches import create_minibatches
from .layers import *
from .loss import *
from .optimizers import *
from .metrics import *
from .history import History
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

def create_net_from_file(filename):
    with open(filename, 'r') as fp:
        info = json.load(fp)

    layers = []
    for layer_info in info["layers"]:
        layer_type = globals()[layer_info['activation']]
        layer = layer_type(units=layer_info['units'])
        layers.append(layer)
    net = Sequential(input_dim=info["input_dim"], layers=layers)

    optimizer = globals()[info["optimizer"]](net, info["lr"])
    loss = globals()[info["loss"]]()
    info_metrics = info["metrics"]
    for metric_name, metric_func in info_metrics.items():
        info_metrics[metric_name] = globals()[metric_func]
    net.compile(loss, optimizer, info_metrics)

    return net

class Model:
    def __init__(self):
        pass


class Sequential(Model):
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
        batch_metrics = {}
        batch_size = len(X)
        
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)
        batch_metrics["loss"] = loss * batch_size
        grad_loss = self.criterion.grad_loss(y_pred, y)
        self.backward(grad_loss)
        self.optimizer.update()
        
        if self.metrics:
            for metric_name, metric_func in self.metrics.items():
                batch_metrics[metric_name] = metric_func(y_pred, y) * batch_size
        
        return batch_metrics
        
    def train_epoch_step(self, X, y, batch_size):
        epoch_metrics = {"loss": 0}
        if self.metrics:
            for metric_name in self.metrics:
                epoch_metrics[metric_name] = 0
        
        m = len(X)
        mini_batches = create_minibatches(X, y, batch_size)
        
        for batch_x, batch_y in mini_batches:    
            batch_metrics = self.train_batch_step(batch_x, batch_y)
            for metric_name, metric_value in batch_metrics.items():
                epoch_metrics[metric_name] += metric_value
        
        #weighted average of batch metrics
        for metric_name in epoch_metrics:
            epoch_metrics[metric_name] /= m
        
        return epoch_metrics

    
    def validation_batch_step(self, X, y):
        batch_metrics = {}
        
        batch_size = len(X)
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)
        batch_metrics["loss"] = loss * batch_size
        
        if self.metrics:
            for metric_name, metric_func in self.metrics.items():
                batch_metrics[metric_name] = metric_func(y_pred, y) * batch_size
        
        return batch_metrics
        
    def validation_epoch_step(self, X, y):
        epoch_metrics = {}
        
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)
        epoch_metrics["loss"] = loss
            
        if self.metrics:
            for metric_name, metric_func in self.metrics.items():
                epoch_metrics[metric_name] = metric_func(y_pred, y)
            
        return epoch_metrics 
    
            
    def fit(self, X, y, epochs=100, batch_size=32, validation=None, validation_data=None, verbose=0, early_stopping=None):
        history = { "train": {
                "loss": [] }
        }
        
        if self.metrics:
            for metric_name in self.metrics:
                history["train"][metric_name] = []
        
        if validation:
            X_cv, y_cv = validation_data
            history["validation"] = {
                "loss": []
            }
            if self.metrics:
                for metric_name in self.metrics:
                    history["validation"][metric_name] = []
        
        for epoch in range(epochs):
            #training
            before_train_time = time.time()
            epoch_metrics = self.train_epoch_step(X, y, batch_size)
            after_train_time = time.time()
            for metric_name, metric_value in epoch_metrics.items():
                history["train"][metric_name].append(metric_value)

            #verbose
            if verbose > 0 and epoch % verbose == 0:
                print(f"Epoch {epoch} | {after_train_time - before_train_time:2f}s | train loss: {history['train']['loss'][-1]}", end="")
                if self.metrics and self.metrics["accuracy"]:
                    print(f" | train accuracy: {history['train']['accuracy'][-1]}")
            
            #validation
            if validation:
                epoch_metrics = self.validation_epoch_step(X_cv, y_cv)
                for metric_name, metric_value in epoch_metrics.items():
                    history["validation"][metric_name].append(metric_value)
            
            #early_stopping
            if early_stopping and early_stopping(history["train"]["loss"][-1]):
                break
        
        return history